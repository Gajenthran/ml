/*!
 * \file main.c
 * \brief Fichier principale concernant 
 * l'application de kNN
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

  config_t * cfg = init_config(CONFIG_FILE);

  data_t * data = NULL, 
         * test = NULL,
         * train = NULL,
         * predicted = NULL;

  data = read_file(argv[1], cfg);
  // normalize(data, cfg);

  const int * sh = init_shuffle(cfg->data_sz);
  test = test_split(data, sh, cfg);
  train = train_split(data, sh, cfg);

  knn_t * knn = NULL;
  knn = init_knn(cfg);
  knn->train = train_split(data, sh, cfg);
  predicted = predict(knn, test, cfg);
  printf("predict score: %.2f\n", predict_score(data, predicted, cfg));

  free_config(cfg);
  free_data(data, train, test);
  free_knn(knn);

#ifdef DEBUG
  print_config(cfg);
  print_data(data, cfg);
#endif

  return 0;
}