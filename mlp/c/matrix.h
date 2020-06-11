/*!
 * \file matrix.h
 * \brief Fichier header de matrix.c
 * \author PANCHALINGAMOORTHY Gajenthran
 */
#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "parser.h"
#include "config.h"

/* Structure représentant une matrice */
typedef struct matrix matrix_t;
struct matrix {
  int rows;       // nombre de lignes
  int cols;       // nombre de colonnes
  double ** data; // données
};

matrix_t * mat_init(int, int);
matrix_t * mat_zinit(int, int);
void       mat_sum(matrix_t *, matrix_t *);
matrix_t * mat_sub(matrix_t *, int);
matrix_t * mat_mul(matrix_t *, matrix_t *);
void       mat_mul_scalar(matrix_t *, double);
void       mat_sigmoid(matrix_t *);
matrix_t * mat_dsigmoid(matrix_t *);
matrix_t * mat_transpose(matrix_t *);
matrix_t * mat_dot(matrix_t *, matrix_t *);
matrix_t * mat_reshape_col(matrix_t * a);
matrix_t * array_to_mat(double *, int);
void       mat_free(matrix_t *);
void       mat_print(matrix_t *);


#endif