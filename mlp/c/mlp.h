/*!
 * \file mlp.h
 * \brief Fichier header de mlp.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _MLP_H_
#define _MLP_H_

#include "parser.h"
#include "config.h"
#include "matrix.h"

/** \brief Structure représentant les données */
typedef struct mlp mlp_t;
struct mlp {
  matrix_t ** w;         // poids du modèle
  matrix_t ** act;		 // état d'activation
  matrix_t ** derivates; // dérivées
  int * layers;          // taille des couches
  double alpha;          // coefficicent d'apprentissage
  int n_layers;          // nombre de couches
  int n_hidden_layers;   // nombre de couches cachées
  int input_sz;          // le nombre d'entrées
  int output_sz;         // le nombre de sorties
};

mlp_t * init_mlp(config_t *);
void    train(mlp_t *, data_t *, config_t *);
void    predict(mlp_t *, data_t *, data_t *, config_t *);
double  mse(mlp_t *, data_t *, data_t *, config_t *);
void    free_mlp(mlp_t * mlp);

#endif