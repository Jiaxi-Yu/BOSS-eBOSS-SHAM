#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include "load_conf.h"

/******************************************************************************
Function `init_conf`:
  Initialize the configuration by invalid values.
Input:
  The structure for configuration.

Arguments:
  * `conf`:     the structure for configuration.
******************************************************************************/
void init_conf(CONF *conf) {
  memset(conf->data, '\0', MAX_BUF);
  memset(conf->rand, '\0', MAX_BUF);
  memset(conf->zlist, '\0', MAX_BUF);
  memset(conf->dbin, '\0', MAX_BUF);
  memset(conf->dd, '\0', MAX_BUF);
  memset(conf->dr, '\0', MAX_BUF);
  memset(conf->rr, '\0', MAX_BUF);
  memset(conf->output, '\0', MAX_BUF);

  conf->dskip = -1;
  conf->dwt = -1;
  conf->daux = -1;
  conf->dsel = -1;
  conf->rskip = -1;
  conf->rwt = -1;
  conf->raux = -1;
  conf->rsel = -1;
  conf->dcnvt = -1;
  conf->rcnvt = -1;
  conf->dbdim = -1;
  conf->dbtype = -1;
  conf->rnum = -1;
  conf->r2prec = -MAX_R2_PREC - 1;
  conf->cmode = -1;
  conf->cfmode = -1;
  conf->moment = -1;
  conf->force = 0;
  conf->verb = -1;

  conf->dx[0] = conf->dx[1] = INIT_DBL;
  conf->dy[0] = conf->dy[1] = INIT_DBL;
  conf->dz[0] = conf->dz[1] = INIT_DBL;
  conf->dw[0] = conf->dw[1] = INIT_DBL;
  conf->da[0] = conf->da[1] = INIT_DBL;
  conf->rx[0] = conf->rx[1] = INIT_DBL;
  conf->ry[0] = conf->ry[1] = INIT_DBL;
  conf->rz[0] = conf->rz[1] = INIT_DBL;
  conf->rw[0] = conf->rw[1] = INIT_DBL;
  conf->ra[0] = conf->ra[1] = INIT_DBL;
  conf->OmegaM = -1;
  conf->rmin = -1;
  conf->rmax = -1;

  conf->flag_dd = 0;
  conf->flag_dr = 0;
  conf->flag_rr = 0;
}


/******************************************************************************
Function `read_opt`:
  Read command line options.
Input:
  Command line options, the string for configuration file, and the structure
  for configuration.

Arguments:
  * `argc`:     number of command line options;
  * `argv`:     array of command line options;
  * `cfile`:    filename of the configuration file;
  * `conf`:     the structure for configuration.
******************************************************************************/
void read_opt(const int argc, char * const argv[], char *cfile, CONF *conf) {
  int opts, idx;
  const char *optstr = 
    "htfc:d:l:w:u:s:y:r:L:W:U:S:Y:m:z:i:j:q:k:a:b:n:p:x:e:o:";
  const struct option long_opts[] = {
    { "help",           no_argument,            NULL,   'h'     },
    { "template",       no_argument,            NULL,   't'     },
    { "force",          no_argument,            NULL,   'f'     },
    { "conf",           required_argument,      NULL,   'c'     },
    { "data",           required_argument,      NULL,   'd'     },
    { "data-header",    required_argument,      NULL,   'l'     },
    { "data-wt-col",    required_argument,      NULL,   'w'     },
    { "data-aux-col",   required_argument,      NULL,   'u'     },
    { "data-select",    required_argument,      NULL,   's'     },
    { "data-x-min",     required_argument,      NULL,   OPT_DXL },
    { "data-x-max",     required_argument,      NULL,   OPT_DXU },
    { "data-y-min",     required_argument,      NULL,   OPT_DYL },
    { "data-y-max",     required_argument,      NULL,   OPT_DYU },
    { "data-z-min",     required_argument,      NULL,   OPT_DZL },
    { "data-z-max",     required_argument,      NULL,   OPT_DZU },
    { "data-wt-min",    required_argument,      NULL,   OPT_DWL },
    { "data-wt-max",    required_argument,      NULL,   OPT_DWU },
    { "data-aux-min",   required_argument,      NULL,   OPT_DAL },
    { "data-aux-max",   required_argument,      NULL,   OPT_DAU },
    { "data-convert",   required_argument,      NULL,   'y'     },
    { "rand",           required_argument,      NULL,   'r'     },
    { "rand-header",    required_argument,      NULL,   'L'     },
    { "rand-wt-col",    required_argument,      NULL,   'W'     },
    { "rand-aux-col",   required_argument,      NULL,   'U'     },
    { "rand-select",    required_argument,      NULL,   'S'     },
    { "rand-x-min",     required_argument,      NULL,   OPT_RXL },
    { "rand-x-max",     required_argument,      NULL,   OPT_RXU },
    { "rand-y-min",     required_argument,      NULL,   OPT_RYL },
    { "rand-y-max",     required_argument,      NULL,   OPT_RYU },
    { "rand-z-min",     required_argument,      NULL,   OPT_RZL },
    { "rand-z-max",     required_argument,      NULL,   OPT_RZU },
    { "rand-wt-min",    required_argument,      NULL,   OPT_RWL },
    { "rand-wt-max",    required_argument,      NULL,   OPT_RWU },
    { "rand-aux-min",   required_argument,      NULL,   OPT_RAL },
    { "rand-aux-max",   required_argument,      NULL,   OPT_RAU },
    { "rand-convert",   required_argument,      NULL,   'Y'     },
    { "omega-m",        required_argument,      NULL,   'm'     },
    { "z-list",         required_argument,      NULL,   'z'     },
    { "bin-dim",        required_argument,      NULL,   'i'     },
    { "bin-type",       required_argument,      NULL,   'j'     },
    { "bin-file",       required_argument,      NULL,   'q'     },
    { "bin-prec",       required_argument,      NULL,   'k'     },
    { "rmin",           required_argument,      NULL,   'a'     },
    { "rmax",           required_argument,      NULL,   'b'     },
    { "rnum",           required_argument,      NULL,   'n'     },
    { "count-mode",     required_argument,      NULL,   'p'     },
    { "moment",         required_argument,      NULL,   'e'     },
    { "cf-mode",        required_argument,      NULL,   'x'     },
    { "dd",             required_argument,      NULL,   OPT_DD  },
    { "dr",             required_argument,      NULL,   OPT_DR  },
    { "rr",             required_argument,      NULL,   OPT_RR  },
    { "output",         required_argument,      NULL,   'o'     },
    { "verbose",        no_argument,    &conf->verb,    1       },
    { "brief",          no_argument,    &conf->verb,    0       },
    { 0, 0, 0, 0}
  };

  opts = idx = 0;
  while ((opts = getopt_long(argc, argv, optstr, long_opts, &idx)) != -1) {
    switch (opts) {
      case 0:
        break;
      case '?':
        P_EXT("please check the command line options.\n");
        exit(ERR_INPUT);
      case 'h':
        usage(argv[0]);
        exit(0);
      case 't':
        temp_conf();
        exit(0);
      case 'f':
        conf->force = 1;
        break;
      case 'c':
        strcpy(cfile, optarg);
        printf("%s\n",cfile);
        break;
      case 'd':
        strcpy(conf->data, optarg);
        break;
      case 'l':
        sscanf(optarg, "%d", &conf->dskip);
        break;
      case 'w':
        sscanf(optarg, "%d", &conf->dwt);
        break;
      case 'u':
        sscanf(optarg, "%d", &conf->daux);
        break;
      case 's':
        sscanf(optarg, "%d", &conf->dsel);
        break;
      case OPT_DXL:
        sscanf(optarg, "%lf", &conf->dx[0]);
        break;
      case OPT_DXU:
        sscanf(optarg, "%lf", &conf->dx[1]);
        break;
      case OPT_DYL:
        sscanf(optarg, "%lf", &conf->dy[0]);
        break;
      case OPT_DYU:
        sscanf(optarg, "%lf", &conf->dy[1]);
        break;
      case OPT_DZL:
        sscanf(optarg, "%lf", &conf->dz[0]);
        break;
      case OPT_DZU:
        sscanf(optarg, "%lf", &conf->dz[1]);
        break;
      case OPT_DWL:
        sscanf(optarg, "%lf", &conf->dw[0]);
        break;
      case OPT_DWU:
        sscanf(optarg, "%lf", &conf->dw[1]);
        break;
      case OPT_DAL:
        sscanf(optarg, "%lf", &conf->da[0]);
        break;
      case OPT_DAU:
        sscanf(optarg, "%lf", &conf->da[1]);
        break;
      case 'y':
        sscanf(optarg, "%d", &conf->dcnvt);
        break;
      case 'r':
        strcpy(conf->rand, optarg);
        break;
      case 'L':
        sscanf(optarg, "%d", &conf->rskip);
        break;
      case 'W':
        sscanf(optarg, "%d", &conf->rwt);
        break;
      case 'U':
        sscanf(optarg, "%d", &conf->raux);
        break;
      case 'S':
        sscanf(optarg, "%d", &conf->rsel);
        break;
      case OPT_RXL:
        sscanf(optarg, "%lf", &conf->rx[0]);
        break;
      case OPT_RXU:
        sscanf(optarg, "%lf", &conf->rx[1]);
        break;
      case OPT_RYL:
        sscanf(optarg, "%lf", &conf->ry[0]);
        break;
      case OPT_RYU:
        sscanf(optarg, "%lf", &conf->ry[1]);
        break;
      case OPT_RZL:
        sscanf(optarg, "%lf", &conf->rz[0]);
        break;
      case OPT_RZU:
        sscanf(optarg, "%lf", &conf->rz[1]);
        break;
      case OPT_RWL:
        sscanf(optarg, "%lf", &conf->rw[0]);
        break;
      case OPT_RWU:
        sscanf(optarg, "%lf", &conf->rw[1]);
        break;
      case OPT_RAL:
        sscanf(optarg, "%lf", &conf->ra[0]);
        break;
      case OPT_RAU:
        sscanf(optarg, "%lf", &conf->ra[1]);
        break;
      case 'Y':
        sscanf(optarg, "%d", &conf->rcnvt);
        break;
      case 'm':
        sscanf(optarg, "%lf", &conf->OmegaM);
        break;
      case 'z':
        strcpy(conf->zlist, optarg);
        break;
      case 'i':
        sscanf(optarg, "%d", &conf->dbdim);
        break;
      case 'j':
        sscanf(optarg, "%d", &conf->dbtype);
        break;
      case 'q':
        strcpy(conf->dbin, optarg);
        break;
      case 'k':
        sscanf(optarg, "%d", &conf->r2prec);
        break;
      case 'a':
        sscanf(optarg, "%lf", &conf->rmin);
        break;
      case 'b':
        sscanf(optarg, "%lf", &conf->rmax);
        break;
      case 'n':
        sscanf(optarg, "%d", &conf->rnum);
        break;
      case 'p':
        sscanf(optarg, "%d", &conf->cmode);
        break;
      case 'e':
        sscanf(optarg, "%d", &conf->moment);
        break;
      case 'x':
        sscanf(optarg, "%d", &conf->cfmode);
        break;
      case OPT_DD:
        strcpy(conf->dd, optarg);
        break;
      case OPT_DR:
        strcpy(conf->dr, optarg);
        break;
      case OPT_RR:
        strcpy(conf->rr, optarg);
        break;
      case 'o':
        strcpy(conf->output, optarg);
        break;
      default:
        break;
    }
  }

  if (optind < argc) {
    P_WRN("unknown command line options:\n ");
    while (optind < argc)
      printf(" %s", argv[optind++]);
    printf("\n");
  }
}


/******************************************************************************
Function `read_conf`:
  Read the configuration file with a format of `KEYWORD = VALUE # COMMENTS`.
Input:
  The filename of the configuration file, and the structure for configuration.
Output:
  Return a non-zero integer if there is problem.

Arguments:
  * `fname`:    the filename of the configuration file;
  * `conf`:     the structure for configuration.
******************************************************************************/
int read_conf(const char *fname, CONF *conf) {
  FILE *fconf;
  char line[MAX_BUF + LEN_KEY];
  char keyword[LEN_KEY];
  char value[MAX_BUF];

  if (!(fconf = fopen(fname, "r"))) {
    P_WRN("cannot open configuration file `%s'.\n", fname);
    return ERR_FILE;
  }
  while (fgets(line, MAX_BUF + LEN_KEY, fconf) != NULL) {
    memset(value, '\0', MAX_BUF);
    sscanf(line, "%*[ |\t]%[^\n]", line);       // Remove leading whitespaces.
    if (line[0] == '#') continue;               // Ignore comments.
    sscanf(line, "%[^=]=%[^\n]", keyword, value);      // Read keyword & value.

    sscanf(keyword, "%s", keyword);             // Remove whitespaces.
    sscanf(value, "%*[ |\t]%[^\n]", value);
    if (value[0] == '\"') {                     // Deal with quotation marks.
      sscanf(value, "%*c%[^\"]", value);
    }
    else {
      sscanf(value, "%[^#]", value);            // Ignore inline comments.
      sscanf(value, "%s", value);               // Remove whitespaces.
    }

    if (!strcmp(keyword, "DATA_CATALOG") && conf->data[0] == '\0')
      strcpy(conf->data, value);
    else if (!strcmp(keyword, "RAND_CATALOG") && conf->rand[0] == '\0')
      strcpy(conf->rand, value);
    else if (!strcmp(keyword, "REDSHIFT_LIST") && conf->zlist[0] == '\0')
      strcpy(conf->zlist, value);
    else if (!strcmp(keyword, "DIST_BIN_FILE") && conf->dbin[0] == '\0')
      strcpy(conf->dbin, value);
    else if (!strcmp(keyword, "DD_FILE") && conf->dd[0] == '\0')
      strcpy(conf->dd, value);
    else if (!strcmp(keyword, "DR_FILE") && conf->dr[0] == '\0')
      strcpy(conf->dr, value);
    else if (!strcmp(keyword, "RR_FILE") && conf->rr[0] == '\0')
      strcpy(conf->rr, value);
    else if (!strcmp(keyword, "OUTPUT") && conf->output[0] == '\0')
      strcpy(conf->output, value);

    else if (!strcmp(keyword, "DATA_HEADER") && conf->dskip == -1)
      sscanf(value, "%d", &conf->dskip);
    else if (!strcmp(keyword, "DATA_WT_COL") && conf->dwt == -1)
      sscanf(value, "%d", &conf->dwt);
    else if (!strcmp(keyword, "DATA_AUX_COL") && conf->daux == -1)
      sscanf(value, "%d", &conf->daux);
    else if (!strcmp(keyword, "DATA_SEL_MODE") && conf->dsel == -1)
      sscanf(value, "%d", &conf->dsel);
    else if (!strcmp(keyword, "RAND_HEADER") && conf->rskip == -1)
      sscanf(value, "%d", &conf->rskip);
    else if (!strcmp(keyword, "RAND_WT_COL") && conf->rwt == -1)
      sscanf(value, "%d", &conf->rwt);
    else if (!strcmp(keyword, "RAND_AUX_COL") && conf->raux == -1)
      sscanf(value, "%d", &conf->raux);
    else if (!strcmp(keyword, "RAND_SEL_MODE") && conf->rsel == -1)
      sscanf(value, "%d", &conf->rsel);
    else if (!strcmp(keyword, "DATA_CONVERT") && conf->dcnvt == -1)
      sscanf(value, "%d", &conf->dcnvt);
    else if (!strcmp(keyword, "RAND_CONVERT") && conf->rcnvt == -1)
      sscanf(value, "%d", &conf->rcnvt);
    else if (!strcmp(keyword, "DIST_BIN_DIM") && conf->dbdim == -1)
      sscanf(value, "%d", &conf->dbdim);
    else if (!strcmp(keyword, "DIST_BIN_TYPE") && conf->dbtype == -1)
      sscanf(value, "%d", &conf->dbtype);
    else if (!strcmp(keyword, "DIST_BIN_RNUM") && conf->rnum == -1)
      sscanf(value, "%d", &conf->rnum);
    else if (!strcmp(keyword, "DIST_BIN_PREC") && conf->r2prec < -MAX_R2_PREC)
      sscanf(value, "%d", &conf->r2prec);
    else if (!strcmp(keyword, "COUNT_MODE") && conf->cmode == -1)
      sscanf(value, "%d", &conf->cmode);
    else if (!strcmp(keyword, "MOMENT") && conf->moment == -1)
      sscanf(value, "%d", &conf->moment);
    else if (!strcmp(keyword, "CF_MODE") && conf->cfmode == -1)
      sscanf(value, "%d", &conf->cfmode);
    else if (!strcmp(keyword, "FORCE") && conf->force != 1)
      sscanf(value, "%d", &conf->force);
    else if (!strcmp(keyword, "VERBOSE") && conf->verb == -1)
      sscanf(value, "%d", &conf->verb);

    else if (!strcmp(keyword, "DATA_X_MIN") && conf->dx[0] >= INIT_DBL)
      sscanf(value, "%lf", &conf->dx[0]);
    else if (!strcmp(keyword, "DATA_X_MAX") && conf->dx[1] >= INIT_DBL)
      sscanf(value, "%lf", &conf->dx[1]);
    else if (!strcmp(keyword, "DATA_Y_MIN") && conf->dy[0] >= INIT_DBL)
      sscanf(value, "%lf", &conf->dy[0]);
    else if (!strcmp(keyword, "DATA_Y_MAX") && conf->dy[1] >= INIT_DBL)
      sscanf(value, "%lf", &conf->dy[1]);
    else if (!strcmp(keyword, "DATA_Z_MIN") && conf->dz[0] >= INIT_DBL)
      sscanf(value, "%lf", &conf->dz[0]);
    else if (!strcmp(keyword, "DATA_Z_MAX") && conf->dz[1] >= INIT_DBL)
      sscanf(value, "%lf", &conf->dz[1]);
    else if (!strcmp(keyword, "DATA_WT_MIN") && conf->dw[0] >= INIT_DBL)
      sscanf(value, "%lf", &conf->dw[0]);
    else if (!strcmp(keyword, "DATA_WT_MAX") && conf->dw[1] >= INIT_DBL)
      sscanf(value, "%lf", &conf->dw[1]);
    else if (!strcmp(keyword, "DATA_AUX_MIN") && conf->da[0] >= INIT_DBL)
      sscanf(value, "%lf", &conf->da[0]);
    else if (!strcmp(keyword, "DATA_AUX_MAX") && conf->da[1] >= INIT_DBL)
      sscanf(value, "%lf", &conf->da[1]);
    else if (!strcmp(keyword, "RAND_X_MIN") && conf->rx[0] >= INIT_DBL)
      sscanf(value, "%lf", &conf->rx[0]);
    else if (!strcmp(keyword, "RAND_X_MAX") && conf->rx[1] >= INIT_DBL)
      sscanf(value, "%lf", &conf->rx[1]);
    else if (!strcmp(keyword, "RAND_Y_MIN") && conf->ry[0] >= INIT_DBL)
      sscanf(value, "%lf", &conf->ry[0]);
    else if (!strcmp(keyword, "RAND_Y_MAX") && conf->ry[1] >= INIT_DBL)
      sscanf(value, "%lf", &conf->ry[1]);
    else if (!strcmp(keyword, "RAND_Z_MIN") && conf->rz[0] >= INIT_DBL)
      sscanf(value, "%lf", &conf->rz[0]);
    else if (!strcmp(keyword, "RAND_Z_MAX") && conf->rz[1] >= INIT_DBL)
      sscanf(value, "%lf", &conf->rz[1]);
    else if (!strcmp(keyword, "RAND_WT_MIN") && conf->rw[0] >= INIT_DBL)
      sscanf(value, "%lf", &conf->rw[0]);
    else if (!strcmp(keyword, "RAND_WT_MAX") && conf->rw[1] >= INIT_DBL)
      sscanf(value, "%lf", &conf->rw[1]);
    else if (!strcmp(keyword, "RAND_AUX_MIN") && conf->ra[0] >= INIT_DBL)
      sscanf(value, "%lf", &conf->ra[0]);
    else if (!strcmp(keyword, "RAND_AUX_MAX") && conf->ra[1] >= INIT_DBL)
      sscanf(value, "%lf", &conf->ra[1]);
    else if (!strcmp(keyword, "OMEGA_M") && conf->OmegaM < 0)
      sscanf(value, "%lf", &conf->OmegaM);
    else if (!strcmp(keyword, "DIST_BIN_RMIN") && conf->rmin < 0)
      sscanf(value, "%lf", &conf->rmin);
    else if (!strcmp(keyword, "DIST_BIN_RMAX") && conf->rmax < 0)
      sscanf(value, "%lf", &conf->rmax);
    else continue;
  }

  fclose(fconf);
  return 0;
}


/******************************************************************************
Function `check_conf`:
  Check the loaded configuration to see if the values are set correctly, and
  the input/output files are valid.
Input:
  The structure for configuration.
Output:
  Return a non-zero integers if there is problem.

Arguments:
  * `conf`:     the structure for configuration.
******************************************************************************/
int check_conf(CONF *conf) {
  int ecode, mode;

  // Check if the parameters are set correctly.
  if (conf->cmode < 0 || conf->cmode > 7) {
    P_ERR("`COUNT_MODE` is not set correctly.\n");
    return ERR_RANGE;
  }

  if (conf->cmode != 4 && conf->cmode != 0) {   // Data catalog is required.
    if ((ecode = check_input(conf->data, "`DATA_CATALOG`")))
      return ecode;

    if (conf->dskip < 0) {
      P_WRN("`DATA_HEADER` is not set correctly.\n"
          "Use the default value (%d) instead.\n", DEFAULT_HEADER);
      conf->dskip = DEFAULT_HEADER;
    }
    if (conf->dwt < 0 || conf->dwt > MAX_COL ||
        (conf->dwt >= 1 && conf->dwt <= 3)) {
      P_WRN("`DATA_WT_COL` is not set correctly.\n"
          "Use the default value (%d) instead.\n", DEFAULT_WT_COL);
      conf->dwt = DEFAULT_WT_COL;
    }
    if (conf->daux < 0 || conf->daux > MAX_COL ||
        (conf->daux >= 1 && conf->daux <= 3)) {
      P_WRN("`DATA_AUX_COL` is not set correctly.\n"
          "Use the default value (%d) instead.\n", DEFAULT_AUX_COL);
      conf->daux = DEFAULT_AUX_COL;
    }
    if (conf->dwt != 0 && conf->daux !=0 && conf->dwt == conf->daux) {
      P_ERR("`DATA_WT_COL` and `DATA_AUX_COL` conflict.\n");
      return ERR_RANGE;
    }
    if (conf->dsel < 0 || conf->dsel > 31) {
      P_WRN("`DATA_SEL_MODE` is not set correctly.\n"
          "Use the default value (%d) instead.\n", DEFAULT_SEL_MODE);
      conf->dsel = DEFAULT_SEL_MODE;
    }

    // Check the selection ranges.
    mode = conf->dsel;
    if (mode % 2 == 1) {        // Selection by X.
      if (conf->dx[1] >= INIT_DBL || conf->dx[0] >= conf->dx[1]) {
        P_ERR("`DATA_X_MIN` or `DATA_X_MAX` is not set correctly.\n");
        return ERR_RANGE;
      }
      mode--;
    }
    if (mode % 4 == 2) {        // Selection by Y.
      if (conf->dy[1] >= INIT_DBL || conf->dy[0] >= conf->dy[1]) {
        P_ERR("`DATA_Y_MIN` or `DATA_Y_MAX` is not set correctly.\n");
        return ERR_RANGE;
      }
      mode -= 2;
    }
    if (mode % 8 == 4) {        // Selection by Z.
      if (conf->dz[1] >= INIT_DBL || conf->dz[0] >= conf->dz[1]) {
        P_ERR("`DATA_Z_MIN` or `DATA_Z_MAX` is not set correctly.\n");
        return ERR_RANGE;
      }
      mode -= 4;
    }
    if (mode % 16 == 8) {       // Selection by WT.
      if (conf->dw[1] >= INIT_DBL || conf->dw[0] >= conf->dw[1]) {
        P_ERR("`DATA_WT_MIN` or `DATA_WT_MAX` is not set correctly.\n");
        return ERR_RANGE;
      }
      if (conf->dwt == 0) {
        P_ERR("non-zero `DATA_WT_COL` is required for `DATA_SEL_MODE` = %d.\n",
            conf->dsel);
        return ERR_RANGE;
      }
      mode -= 8;
    }
    if (mode == 16) {           // Selection by AUX.
      if (conf->da[1] >= INIT_DBL || conf->da[0] >= conf->da[1]) {
        P_ERR("`DATA_AUX_MIN` or `DATA_AUX_MAX` is not set correctly.\n");
        return ERR_RANGE;
      }
      if (conf->daux == 0) {
        P_ERR("non-zero `DATA_AUX_COL` is required for `DATA_SEL_MODE` = %d.\n",
            conf->dsel);
        return ERR_RANGE;
      }
    }
    else {                      // Do not select by AUX
      if (conf->daux) {
        P_WRN("`DATA_AUX_COL` is ignored since `DATA_SEL_MODE` = %d.\n",
            conf->dsel);
        conf->daux = 0;
      }
    }           // Done with checking selection ranges.

    // Check coordinate convention.
    if (conf->dcnvt < 0 || conf->dcnvt > 2) {
      P_WRN("`DATA_CONVERT` is not set correctly.\n"
          "Use the default value (%d) instead.\n", DEFAULT_CONVERT);
      conf->dcnvt = DEFAULT_CONVERT;
    }
    if (conf->dcnvt == 1) {
      if (conf->OmegaM < 0 || conf->OmegaM > 1) {
        P_ERR("`OMEGA_M` is not set correctly.\n");
        return ERR_RANGE;
      }
    }
    else if (conf->dcnvt == 2) {
      if ((ecode = check_input(conf->zlist, "`REDSHIFT_LIST`")))
        return ecode;
    }
  }                     // Done with checking data catalog.

  if (conf->cmode != 1 && conf->cmode != 0) {   // Random catalog is required.
    if ((ecode = check_input(conf->rand, "`RAND_CATALOG`")))
      return ecode;

    if (conf->rskip < 0) {
      P_WRN("`RAND_HEADER` is not set correctly.\n"
          "Use the default value (%d) instead.\n", DEFAULT_HEADER);
      conf->rskip = DEFAULT_HEADER;
    }
    if (conf->rwt < 0 || conf->rwt > MAX_COL ||
        (conf->rwt >= 1 && conf->rwt <= 3)) {
      P_WRN("`RAND_WT_COL` is not set correctly\n"
          "Use the default value (%d) instead.\n", DEFAULT_WT_COL);
      conf->rwt = DEFAULT_WT_COL;
    }
    if (conf->raux < 0 || conf->raux > MAX_COL ||
        (conf->raux >= 1 && conf->raux <= 3)) {
      P_WRN("`RAND_AUX_COL` is not set correctly.\n"
          "Use the default value (%d) instead.\n", DEFAULT_AUX_COL);
      conf->raux = DEFAULT_AUX_COL;
    }
    if (conf->rwt != 0 && conf->raux !=0 && conf->rwt == conf->raux) {
      P_ERR("`RAND_WT_COL` and `RAND_AUX_COL` conflict.\n");
      return ERR_RANGE;
    }
    if (conf->rsel < 0 || conf->rsel > 31) {
      P_WRN("`RAND_SEL_MODE` is not set correctly.\n"
          "Use the default value (%d) instead.\n", DEFAULT_SEL_MODE);
      conf->rsel = DEFAULT_SEL_MODE;
    }

    // Check the selection ranges.
    mode = conf->rsel;
    if (mode % 2 == 1) {        // Selection by X.
      if (conf->rx[1] >= INIT_DBL || conf->rx[0] >= conf->rx[1]) {
        P_ERR("`RAND_X_MIN` or `RAND_X_MAX` is not set correctly.\n");
        return ERR_RANGE;
      }
      mode--;
    }
    if (mode % 4 == 2) {        // Selection by Y.
      if (conf->ry[1] >= INIT_DBL || conf->ry[0] >= conf->ry[1]) {
        P_ERR("`RAND_Y_MIN` or `RAND_Y_MAX` is not set correctly.\n");
        return ERR_RANGE;
      }
      mode -= 2;
    }
    if (mode % 8 == 4) {        // Selection by Z.
      if (conf->rz[1] >= INIT_DBL || conf->rz[0] >= conf->rz[1]) {
        P_ERR("`RAND_Z_MIN` or `RAND_Z_MAX` is not set correctly.\n");
        return ERR_RANGE;
      }
      mode -= 4;
    }
    if (mode % 16 == 8) {       // Selection by WT.
      if (conf->rw[1] >= INIT_DBL || conf->rw[0] >= conf->rw[1]) {
        P_ERR("`RAND_WT_MIN` or `RAND_WT_MAX` is not set correctly.\n");
        return ERR_RANGE;
      }
      if (conf->rwt == 0) {
        P_ERR("non-zero `RAND_WT_COL` is required for `RAND_SEL_MODE` = %d.\n",
            conf->rsel);
        return ERR_RANGE;
      }
      mode -= 8;
    }
    if (mode == 16) {           // Selection by AUX.
      if (conf->ra[1] >= INIT_DBL || conf->ra[0] >= conf->ra[1]) {
        P_ERR("`RAND_AUX_MIN` or `RAND_AUX_MAX` is not set correctly.\n");
        return ERR_RANGE;
      }
      if (conf->raux == 0) {
        P_ERR("non-zero `RAND_AUX_COL` is required for `RAND_SEL_MODE` = %d.\n",
            conf->rsel);
        return ERR_RANGE;
      }
    }
    else {                      // Do not select by AUX.
      if (conf->raux) {
        P_WRN("`RAND_AUX_COL` is ignored since `RAND_SEL_MODE` = %d.\n",
            conf->rsel);
        conf->raux = 0;
      }
    }           // Done with checking selection ranges.

    // Check coordinate convention.
    if (conf->rcnvt < 0 || conf->rcnvt > 2) {
      P_WRN("`RAND_CONVERT` is not set correctly.\n"
          "Use the default value (%d) instead.\n", DEFAULT_CONVERT);
      conf->rcnvt = DEFAULT_CONVERT;
    }
    if (conf->rcnvt == 1 && conf->dcnvt != 1) {
      if (conf->OmegaM < 0 || conf->OmegaM > 1) {
        P_ERR("`OMEGA_M` is not set correctly.\n");
        return ERR_RANGE;
      }
    }
    if (conf->rcnvt == 2 && conf->dcnvt != 2) {
      if ((ecode = check_input(conf->zlist, "`REDSHIFT_LIST`")))
        return ecode;
    }
  }                     // Done with checking random catalog.

  // Check configurations of distance bins.
  if (conf->dbdim <= 0 || conf-> dbdim > 2) {
    P_WRN("`DIST_BIN_DIM` is not set correctly.\n"
        "Use the default value (%d) instead.\n", DEFAULT_BIN_DIM);
    conf->dbdim = DEFAULT_BIN_DIM;
  }

  if (conf->cmode) {    // Distance bins are only required for counting pairs.
    if (conf->dbtype != 0 && conf->dbtype != 1) {
      P_WRN("`DIST_BIN_TYPE` is not set correctly.\n"
          "Use the default value (%d) instead.\n", DEFAULT_BIN_TYPE);
      conf->dbtype = DEFAULT_BIN_TYPE;
    }
    if (conf->dbtype == 0) {    // Read bins from file `BIN_FILE`.
      if ((ecode = check_input(conf->dbin, "`DIST_BIN_FILE`")))
        return ecode;
    }
    else {                      // Set bins using the range and number of bins.
      if (conf->rmin < 0 || conf->rmax < 0 ||
          conf->rmin >= conf->rmax || conf->rnum <= 0) {
        P_ERR("`DIST_BIN_RMIN` or `DIST_BIN_RMAX` or `DIST_BIN_RNUM` "
            "is not set correctly.\n");
        return ERR_RANGE;
      }
    }
    if (conf->dbdim == 2) {
      P_ERR("2-D CF is not supported at this moment.\n");
      return ERR_RANGE;
    }
    if (conf->r2prec < -MAX_R2_PREC || conf->r2prec > MAX_R2_PREC) {
      P_ERR("`DIST_BIN_PREC` is not set correctly.\n");
      return ERR_RANGE;
    }
  }

  // Check outputs.
  mode = conf->cmode;
  if (mode % 2 == 1) {          // DD counts will be computed.
    conf->flag_dd = 1;
    if ((ecode = check_output(conf->dd, "`DD_FILE`", conf->force)))
      return ecode;
    mode--;
  }
  if (mode % 4 == 2) {          // DR counts will be computed.
    conf->flag_dr = 1;
    if ((ecode = check_output(conf->dr, "`DR_FILE`", conf->force)))
      return ecode;
    mode -= 2;
  }
  if (mode == 4) {              // RR counts will be computed.
    conf->flag_rr = 1;
    if ((ecode = check_output(conf->rr, "`RR_FILE`", conf->force)))
      return ecode;
  }

  if (conf->moment != 1 && conf->moment != 2) {
    P_WRN("`MOMENT` is not set correctly.\n"
        "Use the default value (%d) instead.\n", DEFAULT_MOMENT);
    conf->moment = DEFAULT_MOMENT;
  }

  if (conf->cfmode < 0 || conf->cfmode > 3) {
    P_ERR("`CF_MODE` is not set correctly.\n");
    return ERR_RANGE;
  }
  if (conf->cfmode != 0) {
    if ((ecode = check_output(conf->output, "`OUTPUT`", conf->force)))
      return ecode;
  }

  if (conf->flag_dd == 0) {
    if (conf->cfmode == 1 || conf->cfmode == 2 || conf->cfmode == 3) {
      if ((ecode = check_input(conf->dd, "`DD_FILE`"))) return ecode;
      conf->flag_dd = 2;
    }
  }
  if (conf->flag_dr == 0) {
    if (conf->cfmode == 1) {
      if ((ecode = check_input(conf->dr, "`DR_FILE`"))) return ecode;
      conf->flag_dr = 2;
    }
  }
  if (conf->flag_rr == 0) {
    if (conf->cfmode == 1 || conf->cfmode == 2) {
      if ((ecode = check_input(conf->rr, "`RR_FILE`"))) return ecode;
      conf->flag_rr = 2;
    }
  }

  if (conf->verb != 0 && conf->verb != 1) {
    P_WRN("`VERBOSE` is not set correctly.\n"
        "Use the default value (%d) instead.\n", DEFAULT_VERBOSE);
    conf->verb = DEFAULT_VERBOSE;
  }

  return 0;
}


/******************************************************************************
Function `check_input`:
  Check whether the input file can be read.
Input:
  The filename and description of the input file.
Output:
  Return a non-zero integer if there is problem.

Arguments:
  * `fname`:    the filename of the input file;
  * `dscp`:     description of the input file.
******************************************************************************/
int check_input(const char *fname, const char *dscp) {
  if (fname[0] == '\0' || fname[0] == ' ') {
    P_ERR("the input %s is not set.\n", dscp);
    return ERR_FILE;
  }
  if (access(fname, R_OK)) {
    P_ERR("cannot open the %s: `%s'.\n", dscp, fname);
    return ERR_FILE;
  }
  return 0;
}

/******************************************************************************
Function `check_output`:
  Check whether the output file can be written. If the file already exists,
  prompt a choice for overwriting it.
Input:
  The filename and description of the output file.
Output:
  Return a non-zero integer if there is problem.

Arguments:
  * `fname`:    the filename of the output file to be checked;
  * `dscp`:     description of the output file.
  * `force`:    non-zero for overwriting existing files without notifications.
******************************************************************************/
int check_output(const char *fname, const char *dscp, const int force) {
  char confirm, *end;
  char path[MAX_BUF + LEN_KEY];
  int cnt = 0;

  if (fname[0] == '\0' || fname[0] == ' ') {
    P_ERR("the output %s is not set.\n", dscp);
    return ERR_FILE;
  }
  if (!access(fname, F_OK) && force == 0) {     // If the output file exists.
    P_WRN("the output %s `%s' exists.\n", dscp, fname);
    do {
      if ((++cnt) == TIMEOUT) {
        P_ERR("too many failed inputs.\n");
        return ERR_INPUT;
      }
      fprintf(stderr, "Are you going to overwrite it? (y/n): ");
      if (scanf("%c", &confirm) != 1) continue;
      while(getchar() != '\n');         // Ignore invalid inputs.
    }
    while (confirm != 'y' && confirm != 'n');
    if(confirm == 'n') {
      P_ERR("cannot write to the file.\n");
      return ERR_FILE;
    }
  }

  if (!access(fname, F_OK) && access(fname, W_OK)) {
    P_ERR("cannot write to file `%s'.\n", fname);
    return ERR_FILE;
  }

  // Check the path of the output file.
  strcpy(path, fname);
  if ((end = strrchr(path, '/')) != NULL) {
    path[end - &path[0] + 1] = '\0';
    if(access(path, X_OK)) {
      P_ERR("cannot access the path `%s'.\n", path);
      return ERR_FILE;
    }
  }

  return 0;
}


/******************************************************************************
Function `print_conf`:
  Print the relevant items for a given configuration.
Input:
  The structure for configuration.

Arguments:
  * `conf`:     the structure for configuration.
******************************************************************************/
void print_conf(const CONF *conf) {
  int mode;

  printf("\n");
  if (conf->cmode != 4 && conf->cmode != 0) {   // Data catalog is required.
    printf("  DATA_CATALOG    = %s\n", conf->data);
    printf("  DATA_HEADER     = %d\n", conf->dskip);
    if (conf->dwt)
      printf("  DATA_WT_COL     = %d\n", conf->dwt);
    if (conf->daux)
      printf("  DATA_AUX_XOL    = %d\n", conf->daux);
    if (conf->dsel)
      printf("  DATA_SEL_MODE   = %d\n", conf->dsel);

    mode = conf->dsel;
    if (mode % 2 == 1) {        // Selection by X.
      printf("  DATA_X_MIN      = " OFMT_DBL "\n", conf->dx[0]);
      printf("  DATA_X_MAX      = " OFMT_DBL "\n", conf->dx[1]);
      mode--;
    }
    if (mode % 4 == 2) {        // Selection by Y.
      printf("  DATA_Y_MIN      = " OFMT_DBL "\n", conf->dy[0]);
      printf("  DATA_Y_MAX      = " OFMT_DBL "\n", conf->dy[1]);
      mode -= 2;
    }
    if (mode % 8 == 4) {        // Selection by Z.
      printf("  DATA_Z_MIN      = " OFMT_DBL "\n", conf->dz[0]);
      printf("  DATA_Z_MAX      = " OFMT_DBL "\n", conf->dz[1]);
      mode -= 4;
    }
    if (mode % 16 == 8) {       // Selection by WT.
      printf("  DATA_WT_MIN     = " OFMT_DBL "\n", conf->dw[0]);
      printf("  DATA_WT_MAX     = " OFMT_DBL "\n", conf->dw[1]);
      mode -= 8;
    }
    if (mode == 16) {           // Selection by AUX.
      printf("  DATA_AUX_MIN    = " OFMT_DBL "\n", conf->da[0]);
      printf("  DATA_AUX_MAX    = " OFMT_DBL "\n", conf->da[1]);
    }

    if (conf->dcnvt)
      printf("  DATA_CONVERT    = %d\n", conf->dcnvt);
  }             // Done with data catalog.

  if (conf->cmode != 1 && conf->cmode != 0) {   // Random catalog is required.
    printf("  RAND_CATALOG    = %s\n", conf->rand);
    printf("  RAND_HEADER     = %d\n", conf->rskip);
    if (conf->rwt)
      printf("  RAND_WT_COL     = %d\n", conf->rwt);
    if (conf->raux)
      printf("  RAND_AUX_XOL    = %d\n", conf->raux);
    if (conf->rsel)
      printf("  RAND_SEL_MODE   = %d\n", conf->rsel);

    mode = conf->rsel;
    if (mode % 2 == 1) {        // Selection by X.
      printf("  RAND_X_MIN      = " OFMT_DBL "\n", conf->rx[0]);
      printf("  RAND_X_MAX      = " OFMT_DBL "\n", conf->rx[1]);
      mode--;
    }
    if (mode % 4 == 2) {        // Selection by Y.
      printf("  RAND_Y_MIN      = " OFMT_DBL "\n", conf->ry[0]);
      printf("  RAND_Y_MAX      = " OFMT_DBL "\n", conf->ry[1]);
      mode -= 2;
    }
    if (mode % 8 == 4) {        // Selection by Z.
      printf("  RAND_Z_MIN      = " OFMT_DBL "\n", conf->rz[0]);
      printf("  RAND_Z_MAX      = " OFMT_DBL "\n", conf->rz[1]);
      mode -= 4;
    }
    if (mode % 16 == 8) {       // Selection by WT.
      printf("  RAND_WT_MIN     = " OFMT_DBL "\n", conf->rw[0]);
      printf("  RAND_WT_MAX     = " OFMT_DBL "\n", conf->rw[1]);
      mode -= 8;
    }
    if (mode == 16) {           // Selection by AUX.
      printf("  RAND_AUX_MIN    = " OFMT_DBL "\n", conf->ra[0]);
      printf("  RAND_AUX_MAX    = " OFMT_DBL "\n", conf->ra[1]);
    }

    if (conf->rcnvt)
      printf("  RAND_CONVERT    = %d\n", conf->rcnvt);
  }             // Done with random catalog.

  if (conf->dcnvt == 1 || conf->rcnvt == 1)
    printf("  OMEGA_M         = " OFMT_DBL "\n", conf->OmegaM);
  if (conf->dcnvt == 2 || conf->rcnvt == 2)
    printf("  REDSHIFT_LIST   = %s\n", conf->zlist);

  printf("  DIST_BIN_DIM    = %d\n", conf->dbdim);
  if (conf->cmode) {
    printf("  DIST_BIN_TYPE   = %d\n", conf->dbtype);
    if (conf->dbtype) {
      printf("  DIST_BIN_RMIN   = " OFMT_DBL "\n", conf->rmin);
      printf("  DIST_BIN_RMAX   = " OFMT_DBL "\n", conf->rmax);
      printf("  DIST_BIN_RNUM   = %d\n", conf->rnum);
    }
    else
      printf("  DIST_BIN_FILE   = %s\n", conf->dbin);
    printf("  DIST_BIN_PREC   = %d\n", conf->r2prec);
  }

  if (conf->cmode) {
    printf("  COUNT_MODE      = %d\n", conf->cmode);
  }
  printf("  MOMENT          = %d\n", conf->moment);

  if (conf->cfmode) {
    printf("  CF_MODE         = %d\n", conf->cfmode);
    printf("  OUTPUT          = %s\n", conf->output);
  }

  if (conf->flag_dd == 1)               // DD counts will be computed.
    printf("  DD_FILE (OUT)   = %s\n", conf->dd);
  else if (conf->flag_dd == 2)          // DD counts will be read.
    printf("  DD_FILE (IN)    = %s\n", conf->dd);

  if (conf->flag_dr == 1)               // DR counts will be computed.
    printf("  DR_FILE (OUT)   = %s\n", conf->dr);
  else if (conf->flag_dr == 2)          // DR counts will be read.
    printf("  DR_FILE (IN)    = %s\n", conf->dr);

  if (conf->flag_rr == 1)               // RR counts will be computed.
    printf("  RR_FILE (OUT)   = %s\n", conf->rr);
  else if (conf->flag_rr == 2)          // RR counts will be read.
    printf("  RR_FILE (IN)    = %s\n", conf->rr);

  printf("  FORCE           = %d\n", conf->force);
  printf("  VERBOSE         = %d\n", conf->verb);
}


/******************************************************************************
Function `temp_conf`:
  Print a template configuration file.
******************************************************************************/
void temp_conf(void) {
  printf("# Configuration file (default: `%s').\n\
# NOTE that command line options have priority over this file.\n\
# Format: keyword = value # comment\n\
# Use double quotation marks (\") for values with whitespaces (blanks/tabs).\n\
# You do not have to set all the parameters below.\n\
\n\
######################\n\
#  The data catalog  #\n\
######################\n\
DATA_CATALOG    = \n\
        # Input data catalog.\n\
        # Each line of this catalog store the info of one object, where\n\
        # the leading 3 columns are the coordinates.\n\
DATA_HEADER     = 0\n\
        # Number of header lines for the data catalog (default: %d).\n\
DATA_WT_COL     = 0\n\
        # The column for weights in the data catalog (default: %d).\n\
        # The allowed values are:\n\
        # * 0: disable weighting;\n\
        # * 4 - %d: the corresponding column is used for weights.\n\
DATA_AUX_COL    = 0\n\
        # The auxiliary column for object selection of data (deafault: %d).\n\
        # The allowed values are:\n\
        # * 0: disable the auxiliary object selection;\n\
        # * 4 - %d: the corresponding column is used for object selection.\n\
DATA_SEL_MODE   = 0\n\
        # Mode for object selection of data (default: %d).\n\
        # The allowed values are:\n\
        # * 0: disable object selection (use all the data);\n\
        # * 1: select data according to RA or x (the 1st column);\n\
        # * 2: select data according to Dec or y (the 2nd column);\n\
        # * 4: select data according to redshift or z (the 3rd column);\n\
        # * 8: select data according to weight;\n\
        # * 16: select data according to the auxiliary column;\n\
        # * sum of the associated numbers for multi-column selection.\n\
DATA_X_MIN      = 0\n\
DATA_X_MAX      = 0\n\
        # The allowed range of the 1st column for object selection.\n\
DATA_Y_MIN      = 0\n\
DATA_Y_MAX      = 0\n\
        # The allowed range of the 2nd column for object selection.\n\
DATA_Z_MIN      = 0\n\
DATA_Z_MAX      = 0\n\
        # The allowed range of the 3rd column for object selection.\n\
DATA_WT_MIN     = 0\n\
DATA_WT_MAX     = 0\n\
        # The allowed range of the weight column for object selection.\n\
DATA_AUX_MIN    = 0\n\
DATA_AUX_MAX    = 0\n\
        # The allowed range of the auxiliary column for object selection.\n\
\n\
########################\n\
#  The random catalog  #\n\
########################\n\
RAND_CATALOG    = \n\
        # Input random catalog.\n\
        # Each line of this catalog store the info of one object, where\n\
        # the leading 3 columns are the coordinates.\n\
RAND_HEADER     = 0\n\
        # Number of header lines for the random catalog (default: %d).\n\
RAND_WT_COL     = 0\n\
        # The column for weights in the random catalog (default: %d).\n\
        # The allowed values are:\n\
        # * 0: disable weighting;\n\
        # * 4 - %d: the corresponding column is used for weights.\n\
RAND_AUX_COL    = 0\n\
        # The auxiliary column for object selection of random (deafault: %d).\n\
        # The allowed values are:\n\
        # * 0: disable the auxiliary object selection;\n\
        # * 4 - %d: the corresponding column is used for object selection.\n\
RAND_SEL_MODE   = 0\n\
        # Mode for object selection of random (default: %d).\n\
        # The allowed values are:\n\
        # * 0: disable object selection (use all the data);\n\
        # * 1: select data according to RA or x (the 1st column);\n\
        # * 2: select data according to Dec or y (the 2nd column);\n\
        # * 4: select data according to redshift or z (the 3rd column);\n\
        # * 8: select data according to weight;\n\
        # * 16: select data according to the auxiliary column;\n\
        # * sum of the associated numbers for multi-column selection.\n\
RAND_X_MIN      = 0\n\
RAND_X_MAX      = 0\n\
        # The allowed range of the 1st column for object selection.\n\
RAND_Y_MIN      = 0\n\
RAND_Y_MAX      = 0\n\
        # The allowed range of the 2nd column for object selection.\n\
RAND_Z_MIN      = 0\n\
RAND_Z_MAX      = 0\n\
        # The allowed range of the 3rd column for object selection.\n\
RAND_WT_MIN     = 0\n\
RAND_WT_MAX     = 0\n\
        # The allowed range of the weight column for object selection.\n\
RAND_AUX_MIN    = 0\n\
RAND_AUX_MAX    = 0\n\
        # The allowed range of the auxiliary column for object selection.\n\
\n\
##########################\n\
#  Coordinate convention #\n\
##########################\n\
DATA_CONVERT    = 0\n\
RAND_CONVERT    = 0\n\
        # Flag for coordinate convention of objects (default: %d).\n\
        # The allowed values are:\n\
        # * 0: disable coordinate convention;\n\
        # * 1, 2: convert (RA,Dec,redshift) to comoving coordinates based on:\n\
        # ** 1: the matter density (`OMEGA_M`) in a flat LCDM universe;\n\
        # ** 2: a list of redshifts and the corresponding comoving distances,\n\
        #       a cubic spline interpolation is then used to convert redshifts\
\n\
        #       to comoving distances.\n\
OMEGA_M         = \n\
        # The matter density parameter for coordinate convention.\n\
REDSHIFT_LIST   = \n\
        # A file with two columns for redshifts and the corresponding radial\n\
        # comoving distances respectively.\n\
        # The redshift range in this file should cover that of the input\n\
        # catalogs.\n\
        # Lines starting with \"%c\" are omitted as comments.\n\
\n\
###################\n\
#  Distance bins  #\n\
###################\n\
DIST_BIN_DIM    = 1\n\
        # Dimension of the distance bins (default: %d).\n\
        # The allowed values are:\n\
        # * 1: 1-D distance bins for isotropic correlation functions.\n\
        # * 2: 2-D distance bins for 2-D correlation function.\n\
DIST_BIN_TYPE   = 1\n\
        # Method for generating distance bins (default: %d).\n\
        # 0: read distance bins from file `DIST_BIN_FILE`;\n\
        # 1: linear bins set via a range and the number of bins.\n\
DIST_BIN_FILE   = \n\
        # The file for distance bins, each line of the file should be a bin.\n\
        # For 1-D distance bins, the leading 2 columns should be the lower\n\
        # and upper boundaries respectively.\n\
        # Lines starting with \"%c\" are omitted as comments.\n\
DIST_BIN_RMIN   = 0\n\
DIST_BIN_RMAX   = 200\n\
        # Lower and upper limits of the (radial) distance range of interest.\n\
DIST_BIN_RNUM   = 40\n\
        # Number of (radial) distance bins.\n\
DIST_BIN_PREC   = 0\n\
        # Precision of approximate squared distances for the binning.\n\
        # Digits of squared distances after 10^{`R2_PREC`} are truncated.\n\
        # For instance, if the square distance is 1234.5678, and\n\
        # `DIST_BIN_PREC` = 1, then it is truncated to 1230.\n\
        # Only integers in range [-%d,%d] are allowed. If `R2_PREC` is outside\
\n\
        # the range [-%d,%d], then the exact squared distances are used.\n\
        # Note that if the square of boundaries of distance bins are integers,\
\n\
        # the results are still exact if `DIST_BIN_PREC` = 0.\n\
\n\
#############\n\
#  Outputs  #\n\
#############\n\
COUNT_MODE      = 7\n\
        # Things to be counted.\n\
        # The allowed values are:\n\
        # 0: nothing to be computed;\n\
        # 1: DD pair counts;\n\
        # 2: DR pair counts;\n\
        # 4: RR pair counts;\n\
        # Sum of the associated numbers for multiple pair counts.\n\
        # For instance, mode 5 means computing both DD and RR.\n\
MOMENT          = 1\n\
        # Moment of the correlation to be computed (default: %d).\n\
        # The allowed values are:\n\
        # 1: Monopole only (or 2-D correlation function);\n\
        # 2: Both monopole and quadrupole (only for 1-D correlation function).\
\n\
CF_MODE         = 1\n\
        # Estimator for the correlation functions.\n\
        # The allowed values are:\n\
        # 0: Do not compute the correlation functions.\n\
        # 1: Landy & Szalay (1993) estimator ((DD - 2DR + RR) / RR);\n\
        # 2: Natural estimator (DD / RR - 1).\n\
DD_FILE         = \n\
DR_FILE         = \n\
RR_FILE         = \n\
        # The files for DD, DR, and RR pair counts.\n\
        # If the pairs are going to be counted, then this is the output file\n\
        # for counting of pairs; if the pairs will not be counted, but is\n\
        # required by the 2PCF estimator, then this is the input file.\n\
        # The columns of these files are DISTANCE_BIN_RANGE (can be 2 or 4\n\
        # columns), MONOPOLE_COUNT, NORMALIZED_MONOPOLE, QUADRUPOLE_COUNT,\n\
        # NORMALIZED_QUADRUPOLE.\n\
        # Lines starting with \"%c\" are omitted as comments.\n\
OUTPUT          = \n\
        # The output file for the correlation function.\n\
FORCE           = 0\n\
        # Non-zero integer for overwriting existing output files without\n\
        # notifications.\n\
VERBOSE         = 1\n\
        # 0 for concise standard outputs; 1 for detailed outputs (default: %d).\
\n",
      DEFAULT_CONF_FILE,
      DEFAULT_HEADER, DEFAULT_WT_COL, MAX_COL, DEFAULT_AUX_COL, MAX_COL,
      DEFAULT_SEL_MODE,
      DEFAULT_HEADER, DEFAULT_WT_COL, MAX_COL, DEFAULT_AUX_COL, MAX_COL,
      DEFAULT_SEL_MODE,
      DEFAULT_CONVERT, COMMENT, DEFAULT_BIN_DIM, DEFAULT_BIN_TYPE, COMMENT,
      MAX_R2_PREC, MAX_R2_PREC, APPROX_R2_PREC, APPROX_R2_PREC,
      DEFAULT_MOMENT, COMMENT, DEFAULT_VERBOSE);
}


/******************************************************************************
Function `usage`:
  Print the usage of command line options.
Input:
  The name of this program.

Arguments:
  * `pname`:    name of this program.
******************************************************************************/
void usage(char *pname) {
  printf("Usage: %s [OPTION [VALUE]]\n\
Fast Correlation Function Calculator (FCFC).\n\n\
  -h, --help\n\
          Display this message and exit.\n\
  -t, --template\n\
          Display a template configuration and exit.\n\
  -c, --conf=CONF_FILE\n\
          Set the configuration file to CONF_FILE.\n\
          The default configuration file is `%s'.\n\
  -d, --data=DATA_CATALOG\n\
          Set the input data catalog to DATA_CATALOG.\n\
  -r, --rand=RAND_CATALOG\n\
          Set the input random catalog to RAND_CATALOG.\n\
  -l, --data-header=DATA_HEADER\n\
          Skip DATA_HEADER lines when reading the data catalog.\n\
  -L, --rand-header=RAND_HEADER\n\
          Skip RAND_HEADER lines when reading the random catalog.\n\
  -w, --data-wt-col=DATA_WT_COL\n\
          The DATA_WT_COLth column of the data catalog is read as weights.\n\
  -W, --rand-wt-col=RAND_WT_COL\n\
          The RAND_WT_COLth column of the random catalog is read as weights.\n\
  -u, --data-aux-col=DATA_AUX_COL\n\
          The DATA_AUX_COLth column of the data catalog is used for object\n\
          selection.\n\
  -U, --rand-aux-col=RAND_AUX_COL\n\
          The RAND_AUX_COLth column of the random catalog is used for object\n\
          selection.\n\
  -s, --data-select=DATA_SEL_MODE\n\
          Set the data selection method to DATA_SEL_MODE.\n\
  -S, --rand-select=RAND_SEL_MODE\n\
          Set the random selection method to RAND_SEL_MODE.\n\
      --data-x-min=DATA_X_MIN\n\
          Set the lower limit of the 1st column for data selection.\n\
      --data-x-max=DATA_X_MAX\n\
          Set the upper limit of the 1st column for data selection.\n\
      --data-y-min=DATA_Y_MIN\n\
          Set the lower limit of the 2nd column for data selection.\n\
      --data-y-max=DATA_Y_MAX\n\
          Set the upper limit of the 2nd column for data selection.\n\
      --data-z-min=DATA_Z_MIN\n\
          Set the lower limit of the 3rd column for data selection.\n\
      --data-z-max=DATA_Z_MAX\n\
          Set the upper limit of the 3rd column for data selection.\n\
      --data-wt-min=DATA_WT_MIN\n\
          Set the lower limit of the weight for data selection.\n\
      --data-wt-max=DATA_WT_MAX\n\
          Set the upper limit of the weight for data selection.\n\
      --data-aux-min=DATA_AUX_MIN\n\
          Set the lower limit of the auxiliary column for data selection.\n\
      --data-aux-max=DATA_AUX_MAX\n\
          Set the upper limit of the auxiliary column for data selection.\n\
      --rand-x-min=RAND_X_MIN\n\
          Set the lower limit of the 1st column for random selection.\n\
      --rand-x-max=RAND_X_MAX\n\
          Set the upper limit of the 1st column for random selection.\n\
      --rand-y-min=RAND_Y_MIN\n\
          Set the lower limit of the 2nd column for random selection.\n\
      --rand-y-max=RAND_Y_MAX\n\
          Set the upper limit of the 2nd column for random selection.\n\
      --rand-z-min=RAND_Z_MIN\n\
          Set the lower limit of the 3rd column for random selection.\n\
      --rand-z-max=RAND_Z_MAX\n\
          Set the upper limit of the 3rd column for random selection.\n\
      --rand-wt-min=RAND_WT_MIN\n\
          Set the lower limit of the weight for random selection.\n\
      --rand-wt-max=RAND_WT_MAX\n\
          Set the upper limit of the weight for random selection.\n\
      --rand-aux-min=RAND_AUX_MIN\n\
          Set the lower limit of the auxiliary column for random selection.\n\
      --rand-aux-max=RAND_AUX_MAX\n\
          Set the upper limit of the auxiliary column for random selection.\n\
  -y, --data-convert=DATA_CONVERT\n\
          Set the coordinate convention method of the data to DATA_CONVERT.\n\
  -Y, --rand-convert=RAND_CONVERT\n\
          Set the coordinate convention method of the random to RAND_CONVERT.\n\
  -m, --omega-m=OMEGA_M\n\
          Set the matter density parameter for coordinate convention.\n\
  -z, --z-list=REDSHIFT_LIST\n\
          Read the convention of redshift to comoving radial distance from\n\
          file REDSHIFT_LIST.\n\
  -i, --bin-dim=DIST_BIN_DIM\n\
          Set the dimension of distance bins to DIST_BIN_DIM.\n\
  -j, --bin-type=DIST_BIN_TYPE\n\
          Set the method of generating distance bins to DIST_BIN_TYPE.\n\
  -q, --bin-file=DIST_BIN_FILE\n\
          Set the file storing configuration of distance bins to DIST_BIN_FILE.\n\
  -k, --bin-prec=DIST_BIN_PREC\n\
          Set the precision of squared distances to DIST_BIN_PREC.\n\
  -a, --rmin=DIST_BIN_RMIN\n\
          Set the lower limit of the (radial) distance range of interest.\n\
  -b, --rmax=DIST_BIN_RMAX\n\
          Set the upper limit of the (radial) distance range of interest.\n\
  -n, --rnum=DIST_BIN_RNUM\n\
          Set the number of (radial) distance bins to DIST_BIN_RNUM.\n\
  -p, --count-mode=COUNT_MODE\n\
          Set what to be computed for bin counts.\n\
  -e, --moment=MOMENTS\n\
          Set the moments of the correlation to be computed.\n\
  -x, --cf-mode=CF_MODE\n\
          Set the estimator for correlation functions.\n\
      --dd=DD_FILE\n\
          Set the file for DD counts to DD_FILE.\n\
      --dr=DR_FILE\n\
          Set the file for DR counts to DR_FILE.\n\
      --rr=RR_FILE\n\
          Set the file for RR counts to RR_FILE.\n\
  -o, --output=OUTPUT\n\
          Set the output file for the correlation function to `OUTPUT`.\n\
  -f, --force\n\
          Force overwriting existing output files without notifications.\n\
      --verbose\n\
          Display detailed standard outputs.\n\
      --brief\n\
          Display concise standard outputs.\n\
\n\
Consult the -t option for more information on the configuraion.\n\
Report bugs to <zhaocheng03@gmail.com>.\n",
      pname, DEFAULT_CONF_FILE);
}
