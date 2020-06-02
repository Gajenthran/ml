/*!
 * \file main.c
 * \brief Fichier principale concernant 
 * l'application de KMeans
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "parser.h"
#include "config.h"
#include "mlp.h"

void usage(char * exec) {
  fprintf(stderr, "Usage: %s <file>.\n", exec);
  exit(1);
}

int main(int argc, char *argv[]) {
  if(argc != 2)
    usage(argv[0]);

  config_t * cfg = init_config(CONFIG_FILE);

  data_t * data = NULL,
         * train_set = NULL,
         * test_set = NULL;

  data = read_file(argv[1], cfg);
  // normalize(data, cfg);

  const int * sh = init_shuffle(cfg->data_sz);
  test_set = test_split(data, sh, cfg);
  train_set = train_split(data, sh, cfg);

  mlp_t * mlp = init_mlp(cfg);
  train(mlp, train_set, cfg);
  // test(mlp, data, test_set, cfg);


#ifdef DEBUG
  print_config(cfg);
  print_data(data, cfg);
#endif
  
  free_config(cfg);
  free_data(data, train_set, test_set);

  return 0;
}