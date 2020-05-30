/*!
 * \file config.h
 * \brief Fichier header du fichier config.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef __CONFIG_H__
#define __CONFIG_H__

#define CONFIG_FILE "kmeans.cfg"
#define ASCII_AT 64
#define ASCII_BRACE 123

/* Structure représentant la configuration du programme */
typedef struct config config_t;
struct config {
  int nb_val;     // nombre de valeurs dans les données
  int data_sz;    // nombre de données
  int nb_label;   // nombre de labels
  int n_clusters; // nombre de clusters
  int n_iters;    // nombre d'itérations
};

#endif