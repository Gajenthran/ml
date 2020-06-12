/*!
 * \file mlp.c
 * \brief Fichier comprenant les fonctionnalités
 * pour exécuter le modèle de MLP: initialisation
 * et apprentissage.
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mlp.h"

 #define MAX(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


/** \brief Récupère le label le plus probable
 * grâce aux valeurs sorties par le MLP.
 *
 * \param outputs sorties du MLP
 * \param cfg données de configuration
 */
static int get_target(matrix_t * outputs) {
  int i, max_i = 0;
  double max = outputs->data[0][0];

  for(i = 1; i < outputs->cols; i++) {
    if(max < outputs->data[0][i]) {
      max = outputs->data[0][i];
      max_i = i;
    }
  }
  return max_i;
}

/** \brief Initialise les couches du MLP.
 *
 * \param cfg données de configuration
 */
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

/** \brief Initialise les dérivées pour la back
 * propagation du MLP.
 *
 * \param mlp structure de la MLP
 * \param cfg données de configuration
 */
static matrix_t ** init_derivates(mlp_t * mlp, config_t * cfg) {
  matrix_t ** derivates = 
    (matrix_t **)malloc((cfg->n_layers - 1) * sizeof(*derivates));
  assert(derivates);

  int i;
  for(i = 0; i < cfg->n_layers - 1; i++)
    derivates[i] = mat_zinit(mlp->layers[i], mlp->layers[i + 1]);

  return derivates;
}

/** \brief Initialise les états d'activation du
 * MLP.
 *
 * \param mlp structure de la MLP
 * \param cfg données de configuration
 */
static matrix_t ** init_act(mlp_t * mlp, config_t * cfg) {
  matrix_t ** act = (matrix_t **)malloc((cfg->n_layers) * sizeof(*act));
  assert(act);

  int i;
  for(i = 0; i < cfg->n_layers; i++)
    act[i] = mat_zinit(1, mlp->layers[i]);

  return act;
}

/** \brief Initialise les poids du MLP.
 *
 * \param mlp structure de la MLP
 * \param cfg données de configuration
 */
static matrix_t ** init_weights(mlp_t * mlp, config_t * cfg) {
  matrix_t ** w = (matrix_t **)malloc((cfg->n_layers - 1) * sizeof(*w));
  assert(w);

  int i;
  for(i = 0; i < cfg->n_layers - 1; i++)
    w[i] = mat_init(mlp->layers[i], mlp->layers[i + 1]);

  return w;
}

/** \brief Processus de propagation avant pour
 * le MLP.
 *
 * \param mlp structure de la MLP
 * \param data ensemble de données
 */
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

/** \brief Processus de propagation arrière pour
 * le MLP.
 *
 * \param mlp structure de la MLP
 * \param loss taux d'erreur de notre modèle par rapport
 * à la donnée passée
 */
static void back_propagate(mlp_t * mlp, matrix_t * loss) {
  matrix_t * delta = NULL;
  int i;
  for(i = mlp->n_layers - 2; i >= 0; i--) {
    delta = mat_mul(loss, mat_dsigmoid(mlp->act[i + 1]));
    mlp->derivates[i] = mat_dot(mat_reshape_col(mlp->act[i]), delta);
    loss = mat_dot(delta, mat_transpose(mlp->w[i]));
  }
}

/** \brief Applique l'algorithme du gradient
 * pour mettre à jour les poids.
 *
 * \param mlp structure de la MLP
 */
static void gradient_descent(mlp_t * mlp) {
  int i;
  for(i = 0; i < mlp->n_layers - 1; i++) {
    mat_mul_scalar(mlp->derivates[i], mlp->alpha);
    mat_sum(mlp->w[i], mlp->derivates[i]);
  }
}

/** \brief Initialise la structure de MLP.
 *
 * \param cfg données de configuration
 */
mlp_t * init_mlp(config_t * cfg) {
  mlp_t * mlp = (mlp_t *)malloc(sizeof(*mlp));
  assert(mlp);

  mlp->n_layers = cfg->n_layers;
  mlp->n_hidden_layers = cfg->n_hidden_layers;
  mlp->input_sz = cfg->n_val;
  mlp->output_sz = cfg->n_label;
  mlp->alpha = cfg->alpha;

  mlp->layers = init_layers(cfg);
  mlp->w = init_weights(mlp, cfg);
  mlp->derivates = init_derivates(mlp, cfg);
  mlp->act = init_act(mlp, cfg);

  return mlp;
}

/** \brief Phase d'apprentissage des neurones pour le MLP avec une
 * phase de propagation avant et une propagation arrière pour 
 * l'ajustement des poids (algorithme de gradient).
 *
 * \param mlp structure de MLP
 * \param train_set données d'apprentissage
 * \param cfg données de configuration
 */
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

/** \brief Prédire les données tests grâce au MLP
 * entraîné.
 *
 * \param mlp structure de MLP
 * \param data ensemble de données
 * \param test données tests
 * \param cfg données de configuration
 */
void predict(mlp_t * mlp, data_t * data, data_t * test, config_t * cfg) {
  matrix_t * output = NULL;
  int i;

  for(i = 0; i < (int)(cfg->data_sz * cfg->test_size); i++) {  
    output = forward_propagate(mlp, test[i]);
    test[i].target = get_target(output);
  }
}

/** \brief Calcule de l'erreur quadratique moyenne.
 *
 * \param mlp structure de MLP
 * \param data ensemble de données
 * \param test données tests
 * \param cfg données de configuration
 */
double mse(mlp_t * mlp, data_t * data, data_t * test, config_t * cfg) {
  int i, test_sz = (int)(cfg->data_sz * cfg->test_size);
  double errors = 0.0;

  for(i = 0; i < test_sz; i++)
    errors += data[test[i].index].target - test[i].target;

  return pow(errors, 2.0) / test_sz;
}

/** \brief Libère la mémoire de la structure MLP.
 *
 * \param mlp structure de MLP
 */
void free_mlp(mlp_t * mlp) {
  if(mlp) {
    free(mlp);
    mlp = NULL;
  }
}
