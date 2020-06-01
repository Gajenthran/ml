/*!
 * \file knn.h
 * \brief Fichier header de knn.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _KNN_H_
#define _KNN_H_

#include "parser.h"
#include "config.h"

/** \brief Structure représentant les voisins pour le kNN */
typedef struct neighbors neighbors_t;
struct neighbors {
  int index;    // index dans la bdd
  float act;    // état d'activation (distance)
  char * label; // label
};

/** \brief Structure représentant le modèle kNN */
typedef struct knn knn_t;
struct knn {
  data_t * train;          // données d'apprentissage
  neighbors_t * neighbors; // voisins du kNN
  int nb_neighbors;        // nombre de voisins
};

knn_t *  init_knn(config_t *);
data_t * predict(knn_t *, data_t *, config_t *);
double   predict_score(data_t *, data_t *, config_t *);
void     free_knn(knn_t *);

#endif