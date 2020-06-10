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

static int * init_layers(config_t * cfg) {
  int i;
  int * layers = (int *)malloc(cfg->n_layers * sizeof(*layers));
  assert(layers);

  layers[0] = cfg->n_val;
  layers[cfg->n_layers - 1] = cfg->n_label;

  for(i = 1; i < cfg->n_layers - 1; i++)
    layers[i] = cfg->hidden_layers_sz;

  return layers;
}

static matrix_t ** init_derivates(mlp_t * mlp, config_t * cfg) {
  matrix_t ** derivates = 
    (matrix_t **)malloc((cfg->n_layers - 1) * sizeof(*derivates));
  assert(derivates);

  int i;
  for(i = 0; i < cfg->n_layers - 1; i++)
    derivates[i] = mat_zinit(mlp->layers[i], mlp->layers[i + 1]);

  return derivates;
}

static matrix_t ** init_act(mlp_t * mlp, config_t * cfg) {
  matrix_t ** act = (matrix_t **)malloc((cfg->n_layers) * sizeof(*act));
  assert(act);

  int i;
  for(i = 0; i < cfg->n_layers; i++)
    act[i] = mat_zinit(1, mlp->layers[i]);

  return act;
}


static matrix_t ** init_weights(mlp_t * mlp, config_t * cfg) {
  matrix_t ** w = (matrix_t **)malloc((cfg->n_layers - 1) * sizeof(*w));
  assert(w);

  int i;
  for(i = 0; i < cfg->n_layers - 1; i++)
    w[i] = mat_init(mlp->layers[i], mlp->layers[i + 1]);

  return w;
}

mlp_t * init_mlp(config_t * cfg) {
  mlp_t * mlp = (mlp_t *)malloc(sizeof(*mlp));
  assert(mlp);

  mlp->n_layers = cfg->n_layers;
  mlp->n_hidden_layers = cfg->n_hidden_layers;
  mlp->input_sz = cfg->n_val;
  mlp->output_sz = cfg->n_label;
  mlp->hidden_layers_sz = cfg->hidden_layers_sz;
  mlp->alpha = cfg->alpha;

  mlp->layers = init_layers(cfg);
  mlp->w = init_weights(mlp, cfg);
  mlp->derivates = init_derivates(mlp, cfg);
  mlp->act = init_act(mlp, cfg);

  return mlp;
}

static matrix_t * forward_propagate(mlp_t * mlp, data_t data) {
  matrix_t * act = array_to_mat(data.v, mlp->input_sz);
  mlp->act[0] = act;

  int i;
  for(i = 0; i < mlp->n_layers - 1; i++) {
    act = mat_dot(act, mlp->w[i]);
    mat_sigmoid(act);
    mlp->act[i + 1] = act;
  }

  return mlp->act[i];
}

static void back_propagate(mlp_t * mlp, matrix_t * loss) {
  matrix_t * delta = NULL;
  int i;
  for(i = mlp->n_layers - 2; i >= 0; i--) {
    mat_dsigmoid(mlp->act[i + 1]);
    delta = mat_mul(loss, mlp->act[i + 1]);

    mlp->derivates[i] = mat_dot(mat_reshape_col(mlp->act[i]), delta);

    loss = mat_dot(delta, mat_transpose(mlp->w[i]));
  }
}

static void gradient_descent(mlp_t * mlp) {
  int i;
  for(i = 0; i < mlp->n_layers - 1; i++) {
    mat_mul_scalar(mlp->derivates[i], mlp->alpha);
    mat_sum(mlp->w[i], mlp->derivates[i]);
  }
}

void train(mlp_t * mlp, data_t * train_set, config_t * cfg) {
  int i, it, 
      train_sz = cfg->data_sz - (int)(cfg->data_sz * cfg->test_size);

  matrix_t * output, * loss;

  for(it = 0; it < cfg->n_iters; it++) {
    for(i = 0; i < train_sz; i++) {
      output = forward_propagate(mlp, train_set[i]);
      loss = mat_sub(output, train_set[i].target);
      back_propagate(mlp, loss);
      gradient_descent(mlp);
    }
  }
}

/* void test(mlp_t * mlp, data_t * data, data_t * test_set, config_t * cfg) {
  feed_forward(mlp, test_set[0]);
  printf("%s\n", data[test_set[0].index].label);
  mat_print(mlp->layers->out);
} */