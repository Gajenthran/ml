/*!
 * \file kmeans.c
 * \brief Fichier comprenant les fonctionnalités
 * pour exécuter le modèle de KMeans: initialisation
 * et clustering.
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mlp.h"

#define in_w 0
#define out_w ((cfg->n_layers) - 1)


mlp_t * init_mlp(config_t * cfg) {
  mlp_t * mlp = (mlp_t *)malloc(sizeof(*mlp));
  assert(mlp);

  mlp->n_layers = cfg->n_layers;
  mlp->n_hidden_layers = cfg->n_hidden_layers;
  mlp->input_sz = cfg->n_val;
  mlp->output_sz = cfg->n_label;
  mlp->hidden_layers_sz = cfg->hidden_layers_sz;

  mlp->w = (matrix_t **)malloc(cfg->n_layers * sizeof(*mlp->w));
  assert(mlp->w);

  mlp->bias = (matrix_t **)malloc(cfg->n_layers * sizeof(*mlp->bias));
  assert(mlp->bias);

  // input to first hidden layer
  mlp->w[in_w] = mat_init(
    cfg->hidden_layers_sz, cfg->n_val);

  mlp->bias[in_w] = mat_init(
    cfg->hidden_layers_sz, 1);

  // last hidden layer to output
  mlp->w[out_w] = mat_init(
    cfg->n_label, cfg->hidden_layers_sz);

  mlp->bias[out_w] = mat_init(
    cfg->n_label, cfg->hidden_layers_sz);

  // hidden layer to hidden layer
  int i, n_weights = cfg->n_layers - 1;
  for(i = 1; i < n_weights; i++) {
    mlp->w[i] = mat_init(
      cfg->hidden_layers_sz, cfg->hidden_layers_sz);

    mlp->bias[i] = mat_init(
      cfg->hidden_layers_sz, 1);
  }

  return mlp;
}

static matrix_t * feed_forward(mlp_t * mlp, data_t data, config_t * cfg) {
  matrix_t * in = array_to_mat(data.v, cfg->n_val);

  // input to first hidden layer
  matrix_t * h = mat_mul(mlp->w[in_w], in);
  mat_sum(h, mlp->bias[in_w]);
  mat_sigmoid(h);

  // hidden layer to hidden layer
  int i, n_weights = cfg->n_layers - 1;
  for(i = 1; i < n_weights; i++) {
    h = mat_mul(mlp->w[i], h);
    mat_sum(h, mlp->bias[i]);
    mat_sigmoid(h);
  }

  // last hidden layer to output
  matrix_t * out = mat_mul(mlp->w[out_w], h);
  mat_sum(out, mlp->bias[out_w]);
  mat_sigmoid(out);

  mat_free(in);
  mat_free(h);

  return out;
}


void train(mlp_t * mlp, data_t * train_set, config_t * cfg) {
  matrix_t * out = NULL;
  int i, it, 
      train_sz = cfg->data_sz - (int)(cfg->data_sz * cfg->test_size);

  for(it = 0; it < cfg->n_iters; it++) {
    for(i = 0; i < train_sz; i++) {
      out = feed_forward(mlp, train_set[i], cfg);
      // mat_print(out);
    }
  }
}