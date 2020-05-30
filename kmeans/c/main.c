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
#include "kmeans.h"

void usage(char * exec) {
  fprintf(stderr, "Usage: %s <file>.\n", exec);
  exit(1);
}

int main(int argc, char *argv[]) {
  if(argc != 2)
    usage(argv[0]);

  config_t * cfg = init_config(CONFIG_FILE);

  data_t * data = read_file(argv[1], cfg);
  // normalize(data, cfg);

  kmeans_t * kmeans = init_kmeans(data, cfg);
  cluster(kmeans, data, cfg);
  print_cluster(kmeans, data, cfg);

#ifdef DEBUG
  print_config(cfg);
  print_data(data, cfg);
#endif

  return 0;
}