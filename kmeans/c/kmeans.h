/*!
 * \file kmeans.h
 * \brief Fichier header de kmeans.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _KMEANS_H_
#define _KMEANS_H_

#include "parser.h"
#include "config.h"

/* Structure représentant un point dans KMeans */
typedef struct point point_t;
struct point {
  data_t data;    // donnée
  int cluster_id; // identifiant du cluster
};

/* Structure représentant le modèle KMeans */
typedef struct kmeans kmeans_t;
struct kmeans {
  point_t * points;    // les données à clusteriser
  double ** centroids; // centroïdes
  int data_sz;         // nombre de données
  int n_clusters;      // nombre de clusters
};

kmeans_t * init_kmeans(data_t *, config_t *);
void       cluster(kmeans_t *, data_t *, config_t *);
void       print_cluster(kmeans_t *, data_t *, config_t *);
void       free_kmeans(kmeans_t *);

#endif