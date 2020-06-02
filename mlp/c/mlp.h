/*!
 * \file kmeans.h
 * \brief Fichier header de kmeans.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _MLP_H_
#define _MLP_H_

#include "parser.h"
#include "config.h"
#include "matrix.h"

typedef struct layer layer_t;
struct layer {
  matrix_t * in;
  matrix_t * h;
  matrix_t * out;
};

typedef struct mlp mlp_t;
struct mlp {
  layer_t * layers;
  matrix_t ** w;
  matrix_t ** bias;
  double alpha;
  short n_layers;
  short n_hidden_layers;
  short input_sz;
  short output_sz;
  short hidden_layers_sz;
};

mlp_t * init_mlp(config_t *);
void    train(mlp_t *, data_t *, config_t *);
void test(mlp_t *, data_t *, data_t *, config_t *);


#endif