/*!
 * \file ann.c
 * \brief Fichier principale concernant 
 * l'application de SOM
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "parser.h"
#include "knn.h"
#include "config.h"

void usage(char * msg) {
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

int main(int argc, char *argv[]) {
  if(argc != 2)
    usage("Usage: ./knn <file>.");

  config_t * cfg = NULL;
  data_t * data = NULL;
  data_t * train = NULL;
  data_t * test = NULL;
  // knn_t * knn = NULL;

  cfg = init_config(CONFIG_FILE);
  data = read_file(argv[1], cfg);
  normalize(data, cfg);

  train_test_split(data, train, test, cfg);
  // knn = init_knn(data, cfg);

#ifdef DEBUG
  print_config(cfg);
  print_data(data, cfg);
#endif

  return 0;
}