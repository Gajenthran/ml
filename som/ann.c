/*!
 * \file ann.c
 * \brief Fichier principale concernant 
 * l'application de SOM
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "som.h"
#include "parser.h"
#include "config.h"

void usage(char * msg) {
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

int main(int argc, char *argv[]) {
  if(argc != 2)
    usage("Usage: ./som <file>.");

  config_t * cfg = NULL;
  data_t * data = NULL;
  network_t * net = NULL;

  cfg = init_config(CONFIG_FILE);
  data = read_file(argv[1], cfg);
  normalize(data, cfg);
  int * sh = init_shuffle(cfg->data_sz);
  net = init_network(data, cfg);

  train(net, sh, data, cfg);
  label(net, data, cfg);
  print_map(net, cfg);

#ifdef DEBUG
  print_config(cfg);
  print_shuffle(sh, cfg->data_sz);
  print_net(net, cfg);
  print_data(data, cfg);
#endif

  free_config(cfg);
  free_data(data);
  free_network(net);
  return 0;
}