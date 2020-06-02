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

#define in_l 0
#define out_l ((mlp->n_layers) - 1)


mlp_t * init_mlp(config_t * cfg) {
  mlp_t * mlp = (mlp_t *)malloc(sizeof(*mlp));
  assert(mlp);

  mlp->n_layers = cfg->n_layers;
  mlp->n_hidden_layers = cfg->n_hidden_layers;
  mlp->input_sz = cfg->n_val;
  mlp->output_sz = cfg->n_label;
  mlp->hidden_layers_sz = cfg->hidden_layers_sz;
  mlp->alpha = cfg->alpha;

  mlp->layers = (layer_t *)malloc(sizeof(*mlp->layers));
  assert(mlp->layers);

  mlp->w = (matrix_t **)malloc(cfg->n_layers * sizeof(*mlp->w));
  assert(mlp->w);

  mlp->bias = (matrix_t **)malloc(cfg->n_layers * sizeof(*mlp->bias));
  assert(mlp->bias);

  // input to first hidden layer
  mlp->w[in_l] = mat_init(
    cfg->hidden_layers_sz, cfg->n_val);

  mlp->bias[in_l] = mat_init(
    cfg->hidden_layers_sz, 1);

  // last hidden layer to output
  mlp->w[out_l] = mat_init(
    cfg->n_label, cfg->hidden_layers_sz);

  mlp->bias[out_l] = mat_init(
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

static void feed_forward(mlp_t * mlp, data_t data) {
  mlp->layers->in = array_to_mat(data.v, mlp->input_sz);

  // input to first hidden layer
  mlp->layers->h = mat_mul(mlp->w[in_l], mlp->layers->in);
  mat_sum(mlp->layers->h, mlp->bias[in_l]);
  mat_sigmoid(mlp->layers->h);

  // hidden layer to hidden layer
  int i, n_weights = mlp->n_layers - 1;
  for(i = 1; i < n_weights; i++) {
    mlp->layers->h = mat_mul(mlp->w[i], mlp->layers->h);
    mat_sum(mlp->layers->h, mlp->bias[i]);
    mat_sigmoid(mlp->layers->h);
  }

  // last hidden layer to output
  mlp->layers->out = mat_mul(mlp->w[out_l], mlp->layers->h);
  mat_sum(mlp->layers->out, mlp->bias[out_l]);
  mat_sigmoid(mlp->layers->out);
}

static void back_propagation(mlp_t * mlp, data_t data) {
  matrix_t * out_err = mat_sub(mlp->layers->out, data.target);

  matrix_t * gradients = mat_dsigmoid(mlp->layers->out);

  mat_mul_hadamard(gradients, out_err);
  mat_mul_scalar(gradients, mlp->alpha);

  matrix_t * h_trans = mat_transpose(mlp->layers->h);
  matrix_t * w_out_dt = mat_mul(gradients, h_trans);
  
  mat_sum(mlp->w[out_l], w_out_dt);
  mat_sum(mlp->bias[out_l], gradients);

  matrix_t * out_trans = mat_transpose(mlp->w[out_l]);
  matrix_t * h_err = mat_mul(out_trans, out_err);

  matrix_t * h_gradient = mat_dsigmoid(mlp->layers->h);
  mat_mul_hadamard(h_gradient, h_err);
  mat_mul_scalar(h_gradient, mlp->alpha);

  matrix_t * in_trans = mat_transpose(mlp->layers->in);
  matrix_t * w_in_dt = mat_mul(h_gradient, in_trans);

  mat_sum(mlp->w[in_l], w_in_dt);
  mat_sum(mlp->bias[in_l], gradients);
}

void train(mlp_t * mlp, data_t * train_set, config_t * cfg) {
  int i, it, 
      train_sz = cfg->data_sz - (int)(cfg->data_sz * cfg->test_size);

  for(it = 0; it < cfg->n_iters; it++) {
    for(i = 0; i < train_sz; i++) {
      feed_forward(mlp, train_set[i]);
      back_propagation(mlp, train_set[i]);
      feed_forward(mlp, train_set[0]);
      mat_print(mlp->layers->out);

    }
  }
}

void test(mlp_t * mlp, data_t * data, data_t * test_set, config_t * cfg) {
  feed_forward(mlp, test_set[0]);
  printf("%s\n", data[test_set[0].index].label);
  mat_print(mlp->layers->out);
}