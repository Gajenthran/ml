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
#include "kmeans.h"

/** \brief Récupère l'indice du tableau, selon la valeur donnée.
 *
 * \param arr tableau
 * \param idx indice à ne pas vérifier
 * \param val valeur à comparer
 * \param size taille du tableau
 *
 * \return renvoie l'indice du tableau, sinon -1
 */
static int get_index(int * arr, int idx, int val, int size) {
  int i;
  for(i = 0; i < size; i++) {
    if(i == idx) continue;
    if(arr[i] == val) return i;
  }
  return -1;
}

/** \brief Initialise les centroïdes pour le KMeans.
 *
 * \param kmeans modèle KMeans
 * \param data données
 * \param cfg données de configuration
 */
static void init_centroids(kmeans_t * kmeans, data_t * data, config_t * cfg) {
  srand(time(NULL));

  int r, i, c = 0;
  int * centroids = (int *)calloc(0, kmeans->n_clusters * sizeof(*kmeans->centroids));
  assert(centroids);

  kmeans->centroids = (double **)malloc(kmeans->n_clusters * sizeof(*kmeans->centroids));
  assert(kmeans->centroids);

  for(i = 0; i < kmeans->n_clusters; i++) {
    kmeans->centroids[i] = (double *)malloc(cfg->nb_val * sizeof(*kmeans->centroids[i]));
    assert(kmeans->centroids[i]);
  }

  do {
    r = rand() % cfg->data_sz;

    while(1) {
      if(~get_index(centroids, c, r, kmeans->n_clusters)) r = rand() % cfg->data_sz;
      else break;
    }

    centroids[c] = r;
    kmeans->centroids[c] = data[r].v;
    c++;
  } while(c < kmeans->n_clusters);
}

/** \brief Initialise les points représentant les données.
 *
 * \param kmeans modèle KMeans
 * \param data données
 * \param data_sz nombre de données
 */
static void init_points(kmeans_t * kmeans, data_t * data, int data_sz) {
  kmeans->points = (point_t *)malloc(data_sz * sizeof(*kmeans->points));
  assert(kmeans->points);

  int i;
  for(i = 0; i < data_sz; i++) {
    kmeans->points[i].data = data[i];
    kmeans->points[i].cluster_id = -1;
  }
}

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

/** \brief Trouvant le centroïde le plus
 * proche de la donnée.
 *
 * \param kmeans modèle KMeans
 * \param pt point représentant la donnée
 * \param data données
 * \param cfg données de configuration
 *
 * \return le centroïde le plus proche de la donnée.
 */
static int find_cluster(kmeans_t * kmeans, point_t pt, data_t * data, config_t * cfg) {
  int cl, min_cl = 0;
  double dist, min_dist = euclidean_dist(
    kmeans->centroids[0], pt.data.v, cfg->nb_val);

  for(cl = 1; cl < kmeans->n_clusters; cl++) {
    dist = euclidean_dist(kmeans->centroids[cl], pt.data.v, cfg->nb_val);
    if(dist < min_dist) {
      min_dist = dist;
      min_cl = cl;
    }
  }

  return min_cl;
}

/** \brief Met à jour les centroïdes.
 *
 * \param kmeans modèle KMeans
 * \param cfg données de configuration
 */
static void update_centroids(kmeans_t * kmeans, config_t * cfg) {
  int cl, i, d, cluster_sz = 0;
  double sum = 0.0;
  for(cl = 0; cl < kmeans->n_clusters; cl++) {
    for(i = 0; i < cfg->nb_val; i++) {
      sum = 0.0;
      cluster_sz = 0;
      for(d = 0; d < kmeans->data_sz; d++) {
        if(kmeans->points[d].cluster_id == cl) {
          sum += kmeans->points[d].data.v[i];
          cluster_sz++;
        }
      }
      kmeans->centroids[cl][i] = sum / cluster_sz;
    }
  }
}

/** \brief Initialise le modèle KMeans: cluster,
 * centroïdes et points.
 *
 * \param data données
 * \param cfg données de configuration
 *
 * \return le modèle KMeans.
 */
kmeans_t * init_kmeans(data_t * data, config_t * cfg) {
  kmeans_t * kmeans = (kmeans_t *)malloc(sizeof(*kmeans));
  assert(kmeans);

  kmeans->n_clusters = cfg->n_clusters;
  kmeans->data_sz = cfg->data_sz;

  init_centroids(kmeans, data, cfg);
  init_points(kmeans, data, cfg->data_sz);

  return kmeans;
}

void cluster(kmeans_t * kmeans, data_t * data, config_t * cfg) {
  int i, it, cluster_id, clusterized;

  for(it = 0; it < cfg->n_iters; it++) {
    clusterized = 1;
    for(i = 0; i < kmeans->data_sz; i++) {
      cluster_id = find_cluster(kmeans, kmeans->points[i], data, cfg);
      if(kmeans->points[i].cluster_id != cluster_id) {
        kmeans->points[i].cluster_id = cluster_id;
        clusterized = 0;
      }
    }

    if(clusterized)
      break;

    update_centroids(kmeans, cfg);
  }
}

/** \brief Affiche les clusters de KMeans.
 *
 * \param kmeans modèle KMeans
 * \param cfg données de configuration
 */
void print_cluster(kmeans_t * kmeans, data_t * data, config_t * cfg) {
  int i, d;
  for(i = 0; i < kmeans->data_sz; i++) {
    for(d = 0; d < cfg->nb_val; d++)
      printf("%.1f,", kmeans->points[i].data.v[d]);
    printf("Iris-%d (%s)\n", kmeans->points[i].cluster_id, data[i].label);
  }
}

/** \brief Libère le modèle KMeans.
 *
 * \param kmeans modèle KMeans
 */
void free_kmeans(kmeans_t * kmeans) {
  if(kmeans) {
    free(kmeans);
    kmeans = NULL;
  }
}