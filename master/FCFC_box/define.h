/*******************************************************
**                                                    **
**     Common definitions (macros) for FCFC           **
**     Author: Cheng Zhao <zhaocheng03@gmail.com>     **
**                                                    **
*******************************************************/

#ifndef _DEFINE_H_
#define _DEFINE_H_

/******************************************************************************
  Definitions of mathematical/physical constants.
******************************************************************************/
#define PI              3.1415926535
#define SPEED_OF_LIGHT  299792.458

/******************************************************************************
  Definitions for configurations.
******************************************************************************/
#define DEFAULT_CONF_FILE "fcfc.conf"   // Default configuration file.
#define DEFAULT_HEADER    0     // Default number of header lines of inputs.
#define DEFAULT_WT_COL    0     // Default column for weights.
#define DEFAULT_AUX_COL   0     // Default column for object selection.
#define DEFAULT_SEL_MODE  0     // Default object selection mode.
#define DEFAULT_CONVERT   0     // Default flag for converting coordinates.
#define DEFAULT_BIN_DIM   1     // Default dimension of distance bins.
#define DEFAULT_BIN_TYPE  1     // Default type of distance bins.
#define APPROX_R2_PREC    4     // Maximum precision of approximate squared
                                // distances.
#define MAX_R2_PREC       10    // Maximum allowed value for `DIST_BIN_PREC`.
#define DEFAULT_MOMENT    1     // Default moment of correlation.
#define DEFAULT_VERBOSE   1     // Default level of standard outputs.

/******************************************************************************
  Definitions for runtime constants.
******************************************************************************/
#define INIT_DBL 1e30   // Initial (invalid) positive value for double numbers.

#define MAX_BUF 512     // Maximum number of charactors in filenames.
#define LEN_KEY 32      // Maximum number of chars in configuration keywords.
#define TIMEOUT 20      // Maximum number of input trials.
#define BUF     512     // Maximum number of charactors in lines of files.
#define CHUNK   1048576 // Size of the block read each time for line counting.
#define MAX_COL 10      // Maximum number of columns for the input files.
#define DLMT    " ,\t"  // Delimeter for the catalogs.
#define COMMENT '#'     // Comment symbol for input files except the catalogs.
#define FLT_ERR         1e-5    // Error allowed for float number comparison.
#define INTEG_PREC      1e-5    // Precision for numerical integration.
#define NUM_COUNT       0       // Flag for number counts.
#define WT_COUNT        1       // Flag for weight counts.

#define BOXSIZE         2500
//#define BOXSIZE         542.16

/******************************************************************************
  Definitions for the data and data structures.
******************************************************************************/
#define DIM 3                   // Dimension of data.
#define LEAF_SIZE 100           // Maximum number of objects in a leaf node.

/******************************************************************************
  Definitions for the format of outputs.
******************************************************************************/
#define FMT_WARN "\n\x1B[35;1mWarning:\x1B[0m "    // Magenta "Warning".
#define FMT_ERR  "\n\x1B[31;1mError:\x1B[0m "      // Red "Error".
#define FMT_EXIT "\x1B[31;1mExit:\x1B[0m "         // Red "Exit".
#define FMT_DONE "\r\x1B[70C[\x1B[32;1mDONE\x1B[0m]\n"    // Green "DONE".
#define FMT_FAIL "\r\x1B[70C[\x1B[31;1mFAIL\x1B[0m]\n"    // Red "FAIL".
#define OFMT_DBL "%.8g"         // Output format for double numbers.

/******************************************************************************
  Definitions for error codes.
******************************************************************************/
#define ERR_MEM         -1
#define ERR_FILE        -2
#define ERR_RANGE       -3
#define ERR_INPUT       -4
#define ERR_TREE        -5
#define ERR_OTHER       -6

/******************************************************************************
  Definitions for small pieces of codes.
******************************************************************************/
#define P_ERR(...) fprintf(stderr, FMT_ERR __VA_ARGS__)
#define P_WRN(...) fprintf(stderr, FMT_WARN __VA_ARGS__)
#define P_EXT(...) fprintf(stderr, FMT_EXIT __VA_ARGS__)

#define MY_ALLOC(ptr,type,n,message) {                  \
  (ptr) = (type *) malloc(sizeof(type) * (n));          \
  if (!(ptr)) {                                         \
    fprintf(stderr, FMT_ERR #message);                  \
    return ERR_MEM;                                     \
  }                                                     \
}

/******************************************************************************
  Definitions for fundamental data types.
******************************************************************************/
#ifdef DOUBLE_PREC
typedef double real;
#define FMT_REAL "%lf"
#else
typedef float real;
#define FMT_REAL "%f"
#endif

typedef struct {
  real x[DIM];
  real wt;
} DATA;

typedef struct {
  double v[3];
} DOUBLE2;

#endif
