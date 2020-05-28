/*!
 * \file parser.h
 * \brief Fichier header du fichier parser.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _PARSER_H_
#define _PARSER_H_

#include "config.h"

/** \brief Structure représentant les données */
typedef struct data data_t;
struct data {
  double * v;   // vecteur de données
  int index;
  char * label; // étiquette
  double norm;  // norme
};

data_t *   read_file(char *, config_t *);
void       normalize(data_t *, config_t *);
config_t * init_config(char *);
int *      init_shuffle(int size);
void       train_test_split(data_t *, data_t *, data_t *, config_t *);
void    print_train_test(data_t *, data_t *, config_t *);
void       shuffle(int * sh, int size);
void       free_config(config_t *);
void       free_data(data_t * data);
data_t * test_split(data_t * data, config_t * cfg, int * sh);
data_t * train_split(data_t * data, config_t * cfg, int * sh);

#ifdef DEBUG
void       print_data(data_t *, config_t *);
void       print_config(config_t *);
#endif
#endif