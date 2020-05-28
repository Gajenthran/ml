/*!
 * \file knn.c
 * \brief Fichier comprenant les fonctionnalités
 * de parsing pour exécuter le modèle de kNN: 
 * la phase d'apprentissage et de labelisation.
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "knn.h"



knn_t * init_knn(config_t * cfg) {
  knn_t * knn = (knn_t *)malloc(sizeof(*knn));
  assert(knn);

  knn->nb_neighbors = cfg->nb_neighbors;
  knn->neighbors = (neighbors_t *)malloc(
    cfg->nb_neighbors * sizeof(*knn->neighbors));
  assert(knn->neighbors);
  knn->train = NULL;

  return knn;
}

static void init_tab(double * tab, int size) {
  int i;
  for(i = 0; i < size; i++)
    tab[i] = 1.0;
}

static void sort_tab(double * tab, int size) {
  int i, j, min;
  double tmp;
  for(i = 0; i < size-1; i++) {
    min = i;
    for(j = i+1; j < size; j++) {
      if(tab[min] > tab[j]) min = j;
    }
    if(tab[min] != tab[i]) {
      tmp = tab[min];
      tab[min] = tab[i];
      tab[i] = tmp;
    }
  }
}

static const char * label(knn_t * knn, config_t * cfg) {
  int nb_label = cfg->nb_label, l, lab, nbn, * indexes;
  const char * labels[3] = {
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
  };
  indexes = (int *)calloc(0, knn->nb_neighbors * sizeof(*indexes));
  assert(indexes);

  for(nbn = 0; nbn < knn->nb_neighbors; nbn++)
    for(l = 0; l < nb_label; l++)
      if(!strcmp(labels[l], knn->neighbors[nbn].label))
        indexes[l]++;

  for(l = 1; l < nb_label; l++)
    if(indexes[l] > indexes[lab])
      lab = l;
  return labels[lab];
}


data_t * predict(knn_t * knn, data_t * test, config_t * cfg) {
  int i, tr, nbn, tmp_i, tmp_nbn,
      test_size = (int)(cfg->data_sz * cfg->test_size),
      train_size = (cfg->data_sz - test_size);
  double * distances, dist, tmp_dist;
  int * index_distances; 

  distances = (double *)malloc(
    cfg->nb_neighbors * sizeof(*distances));
  assert(distances);

  index_distances = (int *)malloc(
    cfg->nb_neighbors * sizeof(*index_distances));
  assert(index_distances);

  for(i = 0; i < test_size; i++) {
    init_tab(distances, cfg->nb_neighbors);
    for(tr = 0; tr < train_size; tr++) {
      nbn = 0;
      dist = euclidean_dist(
        knn->train[tr].v, test[i].v, cfg->nb_val);
      while(nbn < cfg->nb_neighbors) {
        if(dist < distances[nbn]) {
          tmp_i = index_distances[nbn];
          tmp_dist = distances[nbn];
          distances[nbn] = dist;
          index_distances[nbn] = tr;
          nbn++;
          while(nbn < cfg->nb_neighbors) {
            tmp_i = index_distances[nbn];
            tmp_dist = distances[nbn];
            distances[nbn] = tmp_dist;
            index_distances[nbn] = tmp_i;
            nbn++;
          }
          break;
        }
        nbn++;
      }
    }
    for(nbn = 0; nbn < knn->nb_neighbors; nbn++) {
      knn->neighbors[nbn].act = distances[nbn];
      knn->neighbors[nbn].index = index_distances[nbn];
      knn->neighbors[nbn].label = strdup(knn->train[index_distances[nbn]].label);
    }
    test[i].label = strdup(label(knn, cfg));
  }
  return test;
}


/** \brief Calcule la distance euclidienne de deux vecteurs
 *
 * \param v vecteur v
 * \param w vecteur w
 * \param size taille des vecteurs v et w (partageant la même taille)
 *
 * \return la distance de l'ensemble des deux vecteurs.
 */
double euclidean_dist(double * v, double * w, int size) {
  double sum = 0;
  int i;
  for(i = 0; i < size; i++)
    sum += pow(v[i] - w[i], 2.0);
  return sqrt(sum);
}