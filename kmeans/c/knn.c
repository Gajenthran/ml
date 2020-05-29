/*!
 * \file knn.c
 * \brief Fichier comprenant les fonctionnalités
 * pour exécuter le modèle de kNN: la phase 
 * d'apprentissage et de labelisation.
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "knn.h"

/** \brief Calcule la distance euclidienne de deux vecteurs.
 *
 * \param v vecteur v
 * \param w vecteur w
 * \param size taille des vecteurs v et w (partageant la même taille)
 *
 * \return la distance de l'ensemble des deux vecteurs.
 */
static double euclidean_dist(double * v, double * w, int size) {
  double sum = 0;
  int i;
  for(i = 0; i < size; i++)
    sum += pow(v[i] - w[i], 2.0);
  return sqrt(sum);
}

/** \brief Trouve les k voisins en calculant la distance euclidienne
 * entre le point choisi et les autres données.
 *
 * \param knn structure knn
 * \param test_row donnée à classifier
 * \param distances les distances entre la donnée et les autres
 * \param index_distances les indices des distances entre la donnée et les autres
 */
static void find_neighbors(
  knn_t * knn, data_t test_row, double * distances, int * index_distances, config_t * cfg) {
  int tmp_i, tmp_dist, tr, nbn, 
      train_size = cfg->data_sz - (int)(cfg->data_sz * cfg->test_size);
  double dist;

  for(nbn = 0; nbn < knn->nb_neighbors; nbn++) {
  index_distances[nbn] = nbn;
  distances[nbn] = euclidean_dist(
      knn->train[nbn].v, test_row.v, cfg->nb_val);
  }

  // find k nearest neigbors
  for(tr = 1; tr < train_size; tr++) {
    nbn = 0;
    dist = euclidean_dist(
      knn->train[tr].v, test_row.v, cfg->nb_val);
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

  // assign neighbors
  for(nbn = 0; nbn < knn->nb_neighbors; nbn++) {
    knn->neighbors[nbn].act = distances[nbn];
    knn->neighbors[nbn].index = index_distances[nbn];
    knn->neighbors[nbn].label = strdup(knn->train[index_distances[nbn]].label);
  }
}

/** \brief Labelise les données tests.
 *
 * \param knn structure knn
 * \param cfg données de configuration
 */
static const char * label(knn_t * knn, config_t * cfg) {
  int nb_label = cfg->nb_label, l, lab, nbn, * indexes;

  const char * labels[] = {
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

/** \brief Initialise les distances pour lancer le kNN.
 *
 * \param distances les distances entre la donnée et les autres
 * \param indexes les indices des distances entre la donnée et les autres
 * \param size nombre de distances calculées
 */
static void init_distances(double * distances, int * indexes, int size) {
  int i;
  for(i = 0; i < size; i++) { distances[i] = 1.0; indexes[i] = -1; }
}


/** \brief Initialise le kNN.
 *
 * \param cfg données de configuration
 */
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

/** \brief Prédit la classe des données tests.
 *
 * \param knn structure knn
 * \param test données tests
 * \param cfg données de configuration
 */
data_t * predict(knn_t * knn, data_t * test, config_t * cfg) {
  int i, test_size = (int)(cfg->data_sz * cfg->test_size);

  int * index_distances = (int *)malloc(
    cfg->nb_neighbors * sizeof(*index_distances));
  assert(index_distances);

  double * distances = (double *)malloc(
    cfg->nb_neighbors * sizeof(*distances));
  assert(distances);

  for(i = 0; i < test_size; i++) {
    init_distances(distances, index_distances, cfg->nb_neighbors);
    find_neighbors(knn, test[i], distances, index_distances, cfg);
    test[i].label = strdup(label(knn, cfg));
  }

  free(distances);
  free(index_distances);

  return test;
}

/** \brief Évalue le score de la prédiction.
 *
 * \param data ensemble des données
 * \param test données tests
 * \param cfg données de configuration
 */
double predict_score(data_t * data, data_t * test, config_t * cfg) {
  int i, ds, test_size = (int)(cfg->data_sz * cfg->test_size);
  double rate = 0.0;

  for(i = 0; i < test_size; i++)
    for(ds = 0; ds < cfg->data_sz; ds++)
      if(
        test[i].v[0] == data[ds].v[0] &&
        test[i].v[1] == data[ds].v[1] &&
        test[i].v[2] == data[ds].v[2] &&
        test[i].v[3] == data[ds].v[3]
      ) {
        rate += (!strcmp(test[i].label, data[ds].label)) ? 1.0 : 0;
        break;
      }

  return rate / test_size;
}
