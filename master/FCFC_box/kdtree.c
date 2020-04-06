#include "kdtree.h"


/******************************************************************************
Function `kdtree_build`:
  Construct a k-D tree.
Input:
  A dataset `data` with `ndata` points.
Output:
  Root of the k-D tree;

Arguments:
  * `id`:       the id of the root node, typically 0;
  * `data`:     the input dataset;
  * `ndata`:    number of elements of the input dataset.
******************************************************************************/
KDT* kdtree_build(size_t *id, DATA *data, const size_t ndata) {
  KDT *node;
  int j, dim;
  real mean, tmp, var, var_max;
  size_t i;
  DATA min, max;

  // If there is no data, return NULL/
  if (!data || !ndata) return NULL;

  // Allocate memory for the node.
  node = (KDT *) malloc(sizeof(KDT));
  if (!node) {
    P_EXT("cannot allocate memory for k-D tree construction.\n");
    printf(FMT_FAIL);
    exit(ERR_MEM);
  }

  node->n = ndata;
  node->data = data;
  node->wt = 0;
  node->id = *id;
  (*id)++;

  // Find the dimension with the largest variance, and lower and upper limit
  // of `data`.
  dim = 0;
  var_max = 0;
  for (j = 0; j < DIM; j++) {
    mean = var = 0;
    min.x[j] = max.x[j] = data[0].x[j];

    // Mean and min/max.
    for (i = 0; i < ndata; i++) {
      tmp = data[i].x[j];
      mean += tmp;
      if (min.x[j] > tmp) min.x[j] = tmp;
      if (max.x[j] < tmp) max.x[j] = tmp;
    }
    mean /= (real) ndata;

    for (i = 0; i < ndata; i++) {
      tmp = data[i].x[j] - mean;
      var += tmp * tmp;
    }

    if(var > var_max) {
      dim = j;
      var_max = var;
    }
  }

  node->min = min;
  node->max = max;

  // If the number of points in the dataset is not larger than `LEAF_SIZE`,
  // then create a leaf and return.
  if (ndata <= LEAF_SIZE) {
    node->left = node->right = NULL;
    return node;
  }

  // Split the dataset by the median point in `dim` direction.
  median_split(dim, data, ndata);
  i = ndata >> 1;       // Index of the median point.
  node->left = kdtree_build(id, data, i);
  node->right = kdtree_build(id, data + i, ndata - i);

  return node;
}


/******************************************************************************
Function `kdtree_free`:
  Release the memory allocated for the k-D tree.
Input:
  One node of the tree (typically the root node).

Arguments:
  * `node`:     a pointer to a node of the k-D tree.
******************************************************************************/
void kdtree_free(KDT *node) {
  if(!node) return;
  kdtree_free(node->left);
  kdtree_free(node->right);
  free(node);
}


/******************************************************************************
Function `kdtree_weight`:
  Compute the total weights of the data associated with a node.
Input:
  One node of the k-D tree.

Arguments:
  * `node`:     a pointer to a node of the k-D tree.
******************************************************************************/
double kdtree_weight(KDT *node) {
  size_t i;
  double wt;

  if (node->left == NULL) {
    wt = 0;
    for (i = 0; i < node->n; i++)
      wt += node->data[i].wt;
  }
  else {
    wt = kdtree_weight(node->left) + kdtree_weight(node->right);
  }

  node->wt = wt;
  return wt;
}


/******************************************************************************
Function `kdtree_weight_1`:
  Weight each node with the number of data points, i.e., the weight for every
  data point is 1.
Input:
  One node of the k-D tree.

Arguments:
  * `node`:     a pointer to a node of the k-D tree.
******************************************************************************/
void kdtree_weight_1(KDT *node) {
  if (!node) return;
  kdtree_weight_1(node->left);
  kdtree_weight_1(node->right);
  node->wt = node->n;
}

