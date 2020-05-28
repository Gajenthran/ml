/*!
 * \file knn.h
 * \brief Fichier header de knn.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _KNN_H_
#define _KNN_H_

#include "parser.h"
#include "config.h"

typedef struct neighbors neighbors_t;
struct neighbors {
  int index;
  float act;
  char * label;
};

typedef struct knn knn_t;
struct knn {
  data_t * train;
  neighbors_t * neighbors;
  int nb_neighbors;
};

knn_t * init_knn(config_t *);
void    training(knn_t *, data_t *, config_t *);
double  euclidean_dist(double *, double *, int);
data_t * predict(knn_t * knn, data_t * test, config_t * cfg);
#endif