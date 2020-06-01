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
#include "matrix.h"

#define SIGMOID(x) (1 / (1 + (exp(-(x)))))

matrix_t * mat_init(int rows, int cols) {
  srand(time(NULL));

  matrix_t * mat = (matrix_t *)malloc(sizeof(*mat));
  assert(mat);

  mat->data = (double **)malloc(rows * sizeof(*mat->data));
  assert(mat->data);

  int r, c;
  for(r = 0; r < rows; r++) {
    mat->data[r] = (double *)malloc(cols * sizeof(*mat->data[r]));
    assert(mat->data[r]);

    for(c = 0; c < cols; c++)
      mat->data[r][c] = ((float)rand()/(float)(RAND_MAX)) * 2 - 1;
  }

  mat->rows = rows;
  mat->cols = cols;

  return mat;
}


matrix_t * array_to_mat(double * array, int size) {
  matrix_t * mat = mat_init(size, 1);

  int i;
  for(i = 0; i < size; i++) {
    mat->data[i][0] = array[i];
  }
  return mat;
}

void mat_sum(matrix_t * a, matrix_t * b) {
  int r, c;
  for(r = 0; r < a->rows; r++)
    for(c = 0; c < a->cols; c++)
      a->data[r][c] += b->data[r][c];
}

matrix_t * mat_mul(matrix_t * a, matrix_t * b) {
  if(a->cols != b->rows) {
    fprintf(stderr, "Error: matrix mult.\n");
    exit(1);
  }

  matrix_t * res = mat_init(a->rows, b->cols);

  int i, j, k;
  double sum;
  for(i = 0; i < a->rows; i++) {
    for(j = 0; j < b->cols; j++) {
      sum = 0.0;
      for(k = 0; k < a->cols; k++) {
        sum += a->data[i][k] * b->data[k][j];
      }
      res->data[i][j] = sum;
    }
  }
  return res;
}

matrix_t * mat_transpose(matrix_t * a) {
  matrix_t * res = mat_init(a->cols, a->rows);

  int r, c;
  for(r = 0; r < a->rows; r++)
    for(c = 0; c < a->cols; c++)
      res->data[c][r] = a->data[r][c];

  return res;
}

void mat_sigmoid(matrix_t * a) {
  int r, c;
  for(r = 0; r < a->rows; r++)
    for(c = 0; c < a->cols; c++)
      a->data[r][c] = SIGMOID(a->data[r][c]);
}

void mat_print(matrix_t * mat) {
  int r, c;
  for(r = 0; r < mat->rows; r++) {
    printf("| ");
    for(c = 0; c < mat->cols; c++)
      printf("%.2f ", mat->data[r][c]);
    printf("|\n");
  }
  printf("\n");
}

void mat_free(matrix_t * mat) {
  if(mat) {
    int r;
    for(r = 0; r < mat->rows; r++)
      free(mat->data[r]);
    free(mat->data);
    free(mat);
  }
}










