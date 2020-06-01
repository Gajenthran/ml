/*!
 * \file config.h
 * \brief Fichier header du fichier config.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef __CONFIG_H__
#define __CONFIG_H__

#define CONFIG_FILE "mlp.cfg"
#define ASCII_AT 64
#define ASCII_BRACE 123

/* Structure représentant la configuration du programme */
typedef struct config config_t;
struct config {
  int n_val;            // nombre de valeurs dans les données
  int n_label;          // nombre de labels
  int n_layers;
  int n_hidden_layers;  // nombre de couches cachées
  int hidden_layers_sz; // nombre de couches cachées
  int n_iters;          // nombre d'itérations
  int data_sz;          // nombre de données
  double alpha;         // coefficient d'apprentissage
  double test_size;     // part pour les données tests
};

#endif