#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

typedef double realtype;

#define max(a,b)		((a) > (b) ? (a) : (b))
#define min(a,b)		((a) < (b) ? (a) : (b))

gsl_matrix* moore_penrose_pinv(gsl_matrix *A, const realtype rcond);

void call_pinv(float* pinv_mat, const float* mat, const int NROWS, const int NCOLS);