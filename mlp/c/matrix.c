/*!
 * \file matrix.c
 * \brief Fichier comprenant les fonctionnalités
 * pour gérer une matrice.
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
#define DSIGMOID(y) ((y) * (1 - (y)))

/** \brief Initialise une matrice en mettant
 * les valeurs à 0.
 *
 * \param rows nombre de lignes
 * \param cols nombre de colonnes
 */
matrix_t * mat_zinit(int rows, int cols) {
  matrix_t * mat = (matrix_t *)malloc(sizeof(*mat));
  assert(mat);

  mat->data = (double **)malloc(rows * sizeof(*mat->data));
  assert(mat->data);

  int r, c;
  for(r = 0; r < rows; r++) {
    mat->data[r] = (double *)malloc(cols * sizeof(*mat->data[r]));
    assert(mat->data[r]);

    for(c = 0; c < cols; c++)
      mat->data[r][c] = 0.0;
  }
  mat->rows = rows;
  mat->cols = cols;

  return mat;
}

/** \brief Initialise une matrice en mettant
 * des valeurs aléatoires ([0, 1]).
 *
 * \param rows nombre de lignes
 * \param cols nombre de colonnes
 */
matrix_t * mat_init(int rows, int cols) {
  matrix_t * mat = (matrix_t *)malloc(sizeof(*mat));
  assert(mat);

  mat->data = (double **)malloc(rows * sizeof(*mat->data));
  assert(mat->data);

  int r, c;
  for(r = 0; r < rows; r++) {
    mat->data[r] = (double *)malloc(cols * sizeof(*mat->data[r]));
    assert(mat->data[r]);

    for(c = 0; c < cols; c++) {
      mat->data[r][c] = ((float)rand()/(float)(RAND_MAX));
    }
  }
  mat->rows = rows;
  mat->cols = cols;

  return mat;
}

/** \brief Transforme un vecteur 1D en une matrice.
 *
 * \param array vecteur 1D
 * \param size taille du vecteur
 */
matrix_t * array_to_mat(double * array, int size) {
  matrix_t * mat = mat_init(1, size);

  int i;
  for(i = 0; i < size; i++)
    mat->data[0][i] = array[i];

  return mat;
}

/** \brief Addition matricielle.
 *
 * \param a matrice a
 * \param b matrice b
 */
void mat_sum(matrix_t * a, matrix_t * b) {
  int r, c;
  for(r = 0; r < a->rows; r++)
    for(c = 0; c < a->cols; c++)
      a->data[r][c] += b->data[r][c];
}

/** \brief Soustraction matricielle.
 *
 * \param a matrice
 * \param val valeur pour soustraire la matrice
 */
matrix_t * mat_sub(matrix_t * a, int val) {
  matrix_t * res = mat_init(a->rows, a->cols);
  int r, c;
  for(r = 0; r < a->rows; r++) {
    for(c = 0; c < a->cols; c++) {
      res->data[r][c] = c == val ? 
        1.0 - a->data[r][c] : -a->data[r][c];
    }
  }

  return res;
}

/** \brief Produit matriciel.
 *
 * \param a matrice a
 * \param b matrice b
 */
matrix_t * mat_mul(matrix_t * a, matrix_t * b) {
  matrix_t * res = mat_init(a->rows, b->cols);

  int r, c;
  for(r = 0; r < a->rows; r++)
    for(c = 0; c < b->cols; c++)
      res->data[r][c] = a->data[r][c] * b->data[r][c];
  return res;
}

/** \brief Produit matriciel entre une valeur et une matrice.
 *
 * \param a matrice
 * \param val valeur pour multiplier la matrice
 */
void mat_mul_scalar(matrix_t * a, double val) {
  int r, c;
  for(r = 0; r < a->rows; r++)
    for(c = 0; c < a->cols; c++)
      a->data[r][c] *= val;
}

/** \brief Transposée d'une matrice.
 *
 * \param a matrice
 */
matrix_t * mat_transpose(matrix_t * a) {
  matrix_t * res = mat_init(a->cols, a->rows);

  int r, c;
  for(r = 0; r < a->rows; r++)
    for(c = 0; c < a->cols; c++)
      res->data[c][r] = a->data[r][c];

  return res;
}

/** \brief Applique la fonction sigmoïde sur
 * la matrice.
 *
 * \param a matrice
 */
void mat_sigmoid(matrix_t * a) {
  int r, c;
  for(r = 0; r < a->rows; r++) {
    for(c = 0; c < a->cols; c++) {
      a->data[r][c] = SIGMOID(a->data[r][c]);
    }
  }
}

/** \brief Applique la fonction dérivée de 
 *la sigmoïde sur la matrice.
 *
 * \param a matrice
 */
matrix_t * mat_dsigmoid(matrix_t * a) {
  int r, c;
  matrix_t * res = mat_init(a->rows, a->cols);

  for(r = 0; r < a->rows; r++)
    for(c = 0; c < a->cols; c++)
      res->data[r][c] = DSIGMOID(a->data[r][c]);

  return res;
}

/** \brief Produit scalaire de la matrice.
 *
 * \param a matrice a
 * \param b matrice b
 */
matrix_t * mat_dot(matrix_t * a, matrix_t * b) {
  if(a->cols != b->rows) {
    printf("Erreur: mat_dot\n");
    exit(1);
  }

  matrix_t * res = mat_init(a->rows, b->cols);

  int r, c, k;
  for(r = 0; r < a->rows; r++) {
    for(c = 0; c < b->cols; c++) {
      res->data[r][c] = 0;
      for(k = 0; k < b->rows; k++)
        res->data[r][c] += a->data[r][k] * b->data[k][c];
    }
  }

  return res;
}

/** \brief Redimensionner la matrice en transformant
 * celle-ci sous forme de matrice c*1 où c correspond
 * au nombre de colonnes de la matrice de départ.
 *
 * \param a matrice
 */
matrix_t * mat_reshape_col(matrix_t * a) {
  if(a->rows != 1) {
    printf("Erreur: mat_reshape_col\n");
    exit(1);
  }

  matrix_t * res = mat_init(a->cols, 1);
  int c;
  for(c = 0; c < a->cols; c++) {
    res->data[c][0] = a->data[0][c];
  }

  return res;
}

/** \brief Libère la mémoire de la matrice.
 *
 * \param mat matrice
 */
void mat_free(matrix_t * mat) {
  if(mat) {
    int r;
    for(r = 0; r < mat->rows; r++)
      free(mat->data[r]);
    free(mat->data);
    free(mat);
  }
}

/** \brief Affiche la matrice.
 *
 * \param mat matrice
 */
void mat_print(matrix_t * mat) {
  int r, c;
  printf("rows: %d, cols:%d\n", mat->rows, mat->cols);
  printf("[ \n");
  for(r = 0; r < mat->rows; r++) {
    printf(" [ ");
    for(c = 0; c < mat->cols; c++)
      printf("%.3f, ", mat->data[r][c]);
    printf("]\n");
  }
  printf("]\n\n");
}