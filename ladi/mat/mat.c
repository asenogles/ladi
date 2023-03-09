#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "../pinv/pinv.h"

void matrix_addition(float* A, const float* X, const float* Y, const int N) {
  /* compute element wise addition of two N length arrays (X and Y) into A
     A = X + Y
  */
  for (int i=0; i<N; i++) {
    A[i]=X[i]+Y[i];
  }
}

void inplace_matrix_addition(float* A, const float* B, const int N) {
  /* compute inplace element wise addition of two N length arrays (A and B) into A
     A += B
  */
  for (int i=0; i<N; i++) {
    A[i]+=B[i];
  }
}

void matrix_subtraction(float* A, const float* X, const float* Y, const int N) {
  /* compute element wise subtraction of two N length arrays (Y from X) into A
     A = X - Y
  */
  for (int i=0; i<N; i++) {
    A[i]=X[i]-Y[i];
  }
}

void inplace_matrix_subtraction_AB(float* A, const float* B, const int N) {
  /* compute inplace element wise subtraction of two N length arrays (B from A) into A
     A -= B
  */
  for (int i=0; i<N; i++) {
    A[i]-=B[i];
  }
}

void inplace_matrix_subtraction_BA(float* A, const float* B, const int N) {
  /* compute inplace element wise subtraction of two N length arrays (A from B) into A
     A = B - A
  */
  for (int i=0; i<N; i++) {
    A[i] = B[i] - A[i];
  }
}

void v_dot_square(float* A, const float* X, const float* Y, const int D) {
  /* compute dot product between D length vector and DxD square matrix
     A = X dot Y, where A and Y are D vectors, and X is DxD square matrix
  */
  for (int r=0; r < D; r++) {
    int i = r * D;
    A[r] = 0.f; // Make sure A is zeroed
    for (int c=0; c < D; c++) {
      A[r] += X[i+c] * Y[c];
    }
  }
}

void multi_sum_v_dot_square(float* A, const float* X, const float* Y, const int N, const int D) {
  /* compute dot product between multiple D length vectors and DxD square matrices
     A = X[0].dot(Y[0]) + X[1].dot(Y[1]) .... + X[N].dot(Y[N]) for 0 through N
     Where A is len D vector
     X is N len array of DxD sqaure matrices (NxDxD)
     Y is N len array of len D vectors (NxD)
  */
  
  float* Atmp = (float*)malloc(D * sizeof(float));

  for (int i=0; i<N; i++) {
    int xi = i * D * D;
    int yi = i * D;
    memset(Atmp, 0, D * sizeof(float));
    v_dot_square(Atmp, &X[xi], &Y[yi], D);
    inplace_matrix_addition(A, Atmp, D);
  }

  free(Atmp);
}

void square_matrix_product(float* A, const float* X, const float* Y, const int D) {
  /* compute matrix product between two DxD square matrices
     A = X dot Y, where A, X, & Y = DxD square matrices
  */

  for (int r=0; r < D; r++) {
    int i = r * D;
    for (int j=0; j < D; j++) {
      A[i+j] = 0.f;
      for (int k=0; k < D; k++) {
        A[i+j] += X[i+k] * Y[(k*D)+j]; 
      }
    }
  }
}

void square_matrix_product_t(float* A, const float* X, const float* Y, const int D) {
  /* compute matrix product between a DxD square matrix and the transpose of a second DxD square matrix
     A = X dot Y.T, where A, X, & Y = DxD square matrices
  */

  for (int r=0; r < D; r++) {
    int i = r * D;
    for (int j=0; j < D; j++) {
      A[i+j] = 0.f;
      for (int k=0; k < D; k++) {
        A[i+j] += X[i+k] * Y[(j*D)+k]; 
      }
    }
  }
}

void t_square_matrix_product(float* A, const float* X, const float* Y, const int D) {
  /* compute matrix product between the transpose of a DxD square matrix and a second DxD square matrix
     A = X.T dot Y, where A, X, & Y = DxD square matrices
  */

  for (int r=0; r < D; r++) {
    int i = r * D;
    for (int j=0; j < D; j++) {
      A[i+j] = 0.f;
      for (int k=0; k < D; k++) {
        A[i+j] += X[(k*D)+r] * Y[(k*D)+j]; 
      }
    }
  }
}

void multi_sum_square_product(float* A, const float* X, const float* Y, const int N, const int D) {
  /* compute the sum of matrix products between multiple (length N) DxD square matrices
     A = X[0].dot(Y[0]) + X[1].dot(Y[1]) .... + X[N].dot(Y[N]) for 0 through N
     Where A is DxD sqaure matrix
     X and Y are N len arrays of DxD sqaure matrices (NxDxD)
  */

  float* Atmp = (float*)calloc(D*D, sizeof(float));

  for (int i=0; i<N; i++) {
    int ni = i * D * D;
    square_matrix_product(Atmp, &X[ni], &Y[ni], D);
    inplace_matrix_addition(A, Atmp, D*D);
  }

  free(Atmp);
}

void transpose(float* AT, const float* A, const int D) {
  /* compute the Transpose of a square DxD matrix (A) into AT
     AT = A.T, where A, & AT = DxD square matrices
  */
  for (int r = 0; r < D; r++) {
    int i = r * D;
    for (int j = 0; j < D; j++) {
      AT[i+j] = A[(j*D)+r];
    }
  }
}

void set_square_matrix_identity(float* A, const int D) {
  /* Set a square DxD matrix to the identity matrix
     A = I where A, & I = DxD square matrices
  */
  for (int r=0; r<D; r++) {
    int i = r * D;
    for (int j=0; j<D; j++) {
      if (r == j)
        A[i+j] = 1.f;
      else
        A[i+j] = 0.f;
    }
  }
}

void getCofactor(float* tmp, const float* mat, const int P, const int Q, const int N, const int D) {
  /* Get the co-factor of a matrix (mat)
     tmp = cofactor(mat)
     adapted from https://www.geeksforgeeks.org/determinant-of-a-matrix/
  */

  int i = 0, j = 0;
  // Looping for each element of the matrix
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      // Copying into tmp only elements which are not in given row and column
      if (row != P && col != Q) {
        tmp[(i*D) + j++] = mat[(row*D) + col];
        // Row is filled, so increase row index and reset col index
        if (j == N - 1) {
          j = 0;
          i++;
        }
      }
    }
  }
}

float determinantOfMatrix(const float* mat, int N, const int D) {
  /* Get the determinant of a matrix (mat)
     det = determinant(mat)
     adapted from https://www.geeksforgeeks.org/determinant-of-a-matrix/
  */
  
  // Initialize result
  float det = 0.0f;
 
  // if matrix contains single element
  if (N == 1)
    return mat[0];
 
  float* tmp = malloc(sizeof(float) * (D*D)); // To store cofactors
 
  int sign = 1; // To store sign multiplier
 
  // Iterate for each element of first row
  for (int i = 0; i < N; i++){
    // Getting Cofactor of mat[0,i]
    getCofactor(tmp, mat, 0, i, N, D);
    det += sign * mat[i] * determinantOfMatrix(tmp, N - 1, D);
 
    // terms are to be added with alternate sign
    sign = -sign;
  }
  free(tmp);
  return det;
}

void adjoint(float* adj, const float* mat, const int D) {
  /* Get the adjoint of a matrix (mat)
     adj = adjoint(mat)
     adapted from https://www.geeksforgeeks.org/adjoint-inverse-matrix/
  */

  if (D==1) {
    adj[0] = 1;
    return;
  }

  // temp is used to store cofactors of mat
  float* tmp = malloc(sizeof(float) * (D*D));
  int sign=1;

  for (int i=0; i<D; i++) {
    for (int j=0; j<D; j++) {
      getCofactor(tmp, mat, i, j, D, D);
      sign = ((i+j)%2==0)? 1: -1;
      adj[(j*D)+i] = (sign)*(determinantOfMatrix(tmp, D-1, D));
    }
  }
  free(tmp);
}

void determinant(const float* mat, float* det, const int D) {
  /* Get the determinant of a square DxD matrix (mat)
     det = determinant(mat)
  */
  *det = determinantOfMatrix(mat, D, D);
}

bool inverse_matrix(float* inv, const float* mat, const int D) {
  /* Get the inverse of a square DxD matrix (mat)
     inv = inverse(mat)
     adapted from https://www.geeksforgeeks.org/adjoint-inverse-matrix/
     returns false if pseudo inverse, otherwise true
  */

  // compute determinant
  float det = determinantOfMatrix(mat, D, D);

  // if mat is not invertable, compute pseudo inverse instead
  if (det == 0) {
    // compute pinv using gsl
    call_pinv(inv, mat, D, D);
    return false;
  }

  // compute adjoint matrix
  float* adj = malloc(sizeof(float) * (D*D));
  adjoint(adj, mat, D);

  for (int r=0; r<D; r++) {
    int i = r*D;
    for (int j=0; j<D; j++) {
      inv[i+j] = adj[i+j] / det;
    }
  }
  free(adj);
  return true;
}
