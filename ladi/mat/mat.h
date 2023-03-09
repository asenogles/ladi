#pragma once
#include <stdbool.h> 

void matrix_addition(float* A, const float* X, const float* Y, const int N);

void inplace_matrix_addition(float* A, const float* B, const int N);

void matrix_subtraction(float* A, const float* X, const float* Y, const int N);

void inplace_matrix_subtraction_AB(float* A, const float* B, const int N);

void inplace_matrix_subtraction_BA(float* A, const float* B, const int N);

void v_dot_square(float* A, const float* X, const float* Y, const int D);

void multi_sum_v_dot_square(float* A, const float* X, const float* Y, const int N, const int D);

void square_matrix_product(float* A, const float* X, const float* Y, const int D);

void square_matrix_product_t(float* A, const float* X, const float* Y, const int D);

void t_square_matrix_product(float* A, const float* X, const float* Y, const int D);

void multi_sum_square_product(float* A, const float* X, const float* Y, const int N, const int D);

void set_square_matrix_identity(float* A, const int D);

void getCofactor(float* tmp, const float* mat, const int P, const int Q, const int N, const int D);

float determinantOfMatrix(const float* mat, int N, const int D);

void adjoint(float* adj, const float* mat, const int D);

void determinant(const float* mat, float* det, const int D);

bool inverse_matrix(float* inv, const float* mat, const int D);
