/*!
 * \file parser.c
 * \brief Fichier comprenant les fonctionnalités
 * de parsing pour parser les données en entrée et
 * le fichier config
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "parser.h"

#define QSTRCMP(a, b)  (*(a) != *(b) ? \
    (int) ((unsigned char) *(a) - \
           (unsigned char) *(b)) : \
            strcmp((a), (b)))

/** \brief Lire le fichiers de données, tokenizer son
 * contenu où chaque valeur est séparée par une virgule
 * placer les éléments dans la struct data_t.
 *
 * \param filename nom du fichier
 * \param cfg données de configuration
 *
 * \return la structure de forme data_t qui représente
 * les données formalisées
 */
data_t * read_file(char * filename, config_t * cfg) {
  const int MAX = 1024;
  FILE * fp = fopen(filename, "r");
  if(!fp) {
    fprintf(stderr, "Can't open file %s\n", filename);
    exit(1);
  }

  int line = 0, j = 0;
  char * buf = (char *)malloc(MAX * sizeof(*buf)), * tok, * end;
  assert(buf);
  data_t * data = (data_t *)malloc(MAX * sizeof(*data));
  assert(data);

  while(!feof(fp)) {
    fgets(buf, MAX, fp);
    if(ferror(fp)) {
      fprintf( stderr, "Error while reading file %s\n", filename);
      exit(1);
    }

    // tokenizer la ligne récupérée par fgets
    char * label;
    tok = strtok(buf, ",");
    data[line].v = (double *)malloc(cfg->n_val * sizeof(*data[line].v));
    assert(data[line].v);

    j = 0;
    while(tok != NULL) {
      if(j < cfg->n_val)
        data[line].v[j++] = strtod(tok, &end);
      label = tok;
      tok = strtok(NULL, ",");
    }
    label = strtok(label, "\n");
    if(!strcmp(label, "Iris-setosa")) {
      data[line].target = 0;
    } else if(!strcmp(label, "Iris-versicolor")) {
      data[line].target = 1;
    } else {
      data[line].target = 2;
    }
    data[line++].label = strdup(label);
  }

  cfg->data_sz = line;
  return data;
}

/** \brief Normalise les données.
 *
 * \param data ensemble de données
 * \param cfg  données de configuration
 */
void normalize(data_t * data, config_t * cfg) {
  int i, j;
  double sum;

  for(i = 0; i < cfg->data_sz; i++) {
    sum = 0;
    for(j = 0; j < cfg->n_val; j++)
      sum += pow(data[i].v[j], 2.0);
    data[i].norm = sqrt(sum);
    for(j = 0; j < cfg->n_val; j++)
      data[i].v[j] /= data[i].norm;
  }
}

/** \brief Initialise les données de
 * configuration
 *
 * \param filename fichier de configuration
 * \return la structure de configuration
 * config_t
 */
config_t * init_config(char * filename) {
  const int MAX = 1024;
  FILE * fp = fopen(filename, "r");
  if(!fp) {
    fprintf(stderr, "Can't open file %s\n", filename);
    exit(1);
  }

  char * buf = (char *)malloc(MAX * sizeof(*buf)), * tok, * end;
  assert(buf);
  config_t * cfg = (config_t *)malloc(sizeof *cfg);
  assert(cfg);

  while(!feof(fp)) {
    fgets(buf, MAX, fp);
    if(ferror(fp)) {
      fprintf(stderr, "Error while reading file %s\n", filename);
      exit(1);
    }

    tok = strtok(buf, "=");
    while(tok != NULL) {
      if(
        tok != NULL && 
        tok[0] > ASCII_AT && 
        tok[0] < ASCII_BRACE
      ) {
        if(!QSTRCMP(tok, "N_VAL")) {
          tok = strtok(NULL, "=");
          cfg->n_val = atoi(tok);
        } else if(!QSTRCMP(tok, "N_LABEL")) {
          tok = strtok(NULL, "=");
          cfg->n_label = atoi(tok);
        } else if(!QSTRCMP(tok, "N_HIDDEN_LAYERS")) {
          tok = strtok(NULL, "=");
          cfg->n_hidden_layers = atoi(tok);
        } else if(!QSTRCMP(tok, "HIDDEN_LAYERS_SZ")) {
          tok = strtok(NULL, "=");
          cfg->hidden_layers_sz = atoi(tok);
        } else if(!QSTRCMP(tok, "ALPHA")) {
          tok = strtok(NULL, "=");
          cfg->alpha = strtod(tok, &end);
        } else if(!QSTRCMP(tok, "N_ITERS")) {
          tok = strtok(NULL, "=");
          cfg->n_iters = atoi(tok);
        } else if(!QSTRCMP(tok, "TEST_SIZE")) {
          tok = strtok(NULL, "=");
          cfg->test_size = strtod(tok, &end);
        } else {
          fprintf( stderr, "Error while reading file %s\n", filename);
          exit(1);
        }
      }
      tok = strtok(NULL, "=");
    }
  }

  // layers = hidden + input + output
  cfg->n_layers = cfg->n_hidden_layers + 2;
  return cfg;
}

/** \brief Couper l'ensemble des données pour former les données
 * d'apprentissage.
 *
 * \param data ensemble de données
 * \param sh vecteur représentant l'ordre de passage des données
 * \param cfg données de configuration
 *
 * \return la structure de forme data_t qui représente
 * les données d'apprentissage
 */
data_t * train_split(data_t * data, const int * sh, config_t * cfg) {
  int i, d,
      test_size = (int)(cfg->data_sz * cfg->test_size),
      train_size = cfg->data_sz - test_size;

  data_t * train = (data_t *)malloc(train_size * sizeof(*train));
  assert(train);

  for(i = 0; i < train_size; i++) {
    train[i].label = strdup(data[sh[i + test_size]].label);
    train[i].v = (double *)malloc(cfg->n_val * sizeof(*train[i].v));
    assert(train[i].v);
    for(d = 0; d < cfg->n_val; d++)
      train[i].v[d] = data[sh[i + test_size]].v[d];
    train[i].target = data[sh[i + test_size]].target;
  }

  return train;
}

/** \brief Couper l'ensemble des données pour former les données
 * tests.
 *
 * \param data ensemble de données
 * \param sh vecteur représentant l'ordre de passage des données
 * \param cfg données de configuration
 *
 * \return la structure de forme data_t qui représente
 * les données tests
 */
data_t * test_split(data_t * data, const int * sh, config_t * cfg) {
  int i, d;
  int test_size = (int)(cfg->data_sz * cfg->test_size);
  data_t * test = (data_t *)malloc(test_size * sizeof(*test));
  assert(test);

  for(i = 0; i < test_size; i++) {
    test[i].index = sh[i];
    test[i].v = (double *)malloc(cfg->n_val * sizeof(*test[i].v));
    assert(test[i].v);
    for(d = 0; d < cfg->n_val; d++) {
      test[i].v[d] = data[sh[i]].v[d];
    }
  }

  return test;
}

/** \brief Mélange le vecteur représentant l'ordre de passage des données
 * lors de la phase d'apprentissage
 *
 * \param sh   vecteur représentant l'ordre de passage des données
 * \param size taille du vecteur
 */
static void shuffle(int * sh, int size) {
  int i, r;
  for(i = 0; i < size; i++) {
    r = rand() % size;
    if(sh[i] != sh[r]) {
      sh[i] ^= sh[r];
      sh[r] ^= sh[i];
      sh[i] ^= sh[r];
    }
  }
}

/** \brief Initialise le vecteur représentant l'ordre
 * de passage des données lors de la phase d'apprentissage.
 *
 * \param size nombre de données
 *
 * \return vecteur représentant l'ordre de passage des données.
 */
int * init_shuffle(int size) {
  srand(time(NULL));
  int i;
  int * sh = (int *)malloc(size * sizeof(*sh));
  assert(sh);

  for(i = 0; i < size; i++)
    sh[i] = i;
  shuffle(sh, size);
  return sh;
}

/** \brief Libère les données de configuration.
 *
 * \param cfg données de configuration
 */
void free_config(config_t * cfg) {
  if(cfg) {
    free(cfg);
    cfg = NULL;
  }
}

/** \brief Libère les données de la bd.
 *
 * \param data ensemble de données
 * \param train ensemble de données d'apprentissage
 * \param test ensemble de données tests
 */
void free_data(data_t * data, data_t * train, data_t * test) {
  if(data)  { free(data);  data = NULL;  }
  if(train) { free(train); train = NULL; }
  if(test)  { free(test);  test = NULL;  }
}

#ifdef DEBUG
void print_data(data_t * data, config_t * cfg) {
  int i, j;
  for(i = 0; i < cfg->data_sz; i++) {
    for(j = 0; j < cfg->n_val; j++) {
      printf("%.1f,", data[i].v[j]);
    }
    printf("%s\n", data[i].label);
  }
}

void print_config(config_t * cfg) {
  printf("n_val:           %d\n", cfg->n_val);
  printf("data_sz:         %d\n", cfg->data_sz);
  printf("n_label:         %d\n", cfg->n_label);
  printf("n_iters:         %d\n", cfg->n_iters);
  printf("n_hidden_layers: %d\n", cfg->n_hidden_layers);
  printf("alpha:           %.2f\n", cfg->alpha);
  printf("test_size:       %.2f\n", cfg->test_size);
}
#endif