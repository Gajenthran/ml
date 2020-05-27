/*!
 * \file som.h
 * \brief Fichier header de som.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _KNN_H_
#define _KNN_H_

#include "parser.h"
#include "config.h"

typedef struct point point_t;
struct point {
  int x, y;
};

typedef struct knn knn_t;
struct knn {
  point_t * pts;
  int neighbors;
};

// network_t * init_network(data_t *, config_t *);
/* bmu_t       find_bmu(network_t *, double *, config_t *); */
double      euclidean_dist(double *, double *, int);

#ifdef DEBUG
void        print_net(network_t *, config_t *);
void        print_shuffle(int *, int);
#endif

#endif