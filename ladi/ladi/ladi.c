#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
# include <stdio.h>
#include <math.h>
#include "../mat/mat.h"

void prediction_1st_order(float* Xp, const float* Xi, const float* A, const int D) {
  /* predict the updated state using the 1st order derivative
     Xp = A dot X
  */
  v_dot_square(Xp, A, Xi, D);
}

void predicted_covariance(float* Pp, const float* Pi, const float* A, const float Q, const int D) {
  /* Predict the updated state covariance
     Pp = A dot Pi dot A.T
  */

  // create tmp array
  float* Pi_AT = (float*)malloc(sizeof(float) * (D*D));

  // Compute Pi dot A.T
  square_matrix_product_t(Pi_AT, Pi, A, D);
  
  // Compute A dot P_AT + Q
  for (int r=0; r < D; r++) {
    int i = r * D;
    for (int j=0; j < D; j++) {
      Pp[i+j] = 0.f;
      for (int k=0; k < D; k++) {
        Pp[i+j] += A[i+k] * Pi_AT[(k*D)+j]; 
      }
      // Add Q (process noise)
      Pp[i+j]+=Q;
    }
  }
  
  free(Pi_AT);
}

void kalman_gain(float* Kg, const float* P, const float* R, const float* H, const int D) {
  /* Compute the Kalman gain matrix
     Kg = P.dot(H.T).dot(np.linalg.pinv((H.dot(P).dot(H.T) + R)))
  */

  // Compute P_HT = P dot H.T
  float* P_HT = (float*)malloc(sizeof(float) * D*D);
  square_matrix_product_t(P_HT, P, H, D);

  // Compute H_P_HT = H dot P_HT
  float* H_P_HT = (float*)malloc(sizeof(float) * D*D);
  square_matrix_product(H_P_HT, H, P_HT, D);

  // Compute H_P_HT_R =  H_P_HT + R
  float* H_P_HT_R = (float*)malloc(sizeof(float) * D*D);
  matrix_addition(H_P_HT_R, H_P_HT, R, D*D);

  // Compute H_P_HT_R_INV = pinv(H_P_HT_R)
  float* H_P_HT_R_INV = (float*)malloc(sizeof(float) * D*D);
  inverse_matrix(H_P_HT_R_INV, H_P_HT_R, D);

  // Compute HT_H_P_HT_R_INV = H.T dot H_P_HT_R_INV
  float* HT_H_P_HT_R_INV = (float*)malloc(sizeof(float) * D*D);
  t_square_matrix_product(HT_H_P_HT_R_INV, H, H_P_HT_R_INV, D);

  // Compute Kg = P dot HT_H_P_HT_R_INV
  square_matrix_product(Kg, P, HT_H_P_HT_R_INV, D);

  free(P_HT);
  free(H_P_HT);
  free(H_P_HT_R);
  free(H_P_HT_R_INV);
  free(HT_H_P_HT_R_INV);
}

void update_state(float* X, const float* Xp, const float* Y, const float* Kg, const int D) {
  /* Update the state matrix based on the measurement and predicted state
     X = Xp + Kg dot (Y - [I dot Xp])
  */

  // Y - Xp
  float* Y_Xp = (float*)malloc(sizeof(float) * D);
  matrix_subtraction(Y_Xp, Y, Xp, D);

  // Kg dot (Y - Xp)
  float* Kg_Y_Xp = (float*)malloc(sizeof(float) * D);
  v_dot_square(Kg_Y_Xp, Kg, Y_Xp, D);

  // Xp + Kg dot (Y - Xp)
  matrix_addition(X, Xp, Kg_Y_Xp, D);
  
  free(Y_Xp);
  free(Kg_Y_Xp);
}

void update_state_normed(float* X, const float* Xp, const float* Y, const float* Kg, const int D, const float THRES) {
  /* Update the state matrix based on the denormed measurement and predicted state
     Ydiff = Y[0] - Xp[0]
     Yi = Xp[0] + (Kg[0,0] * Ydiff) + (Kg[1,0] * Ydiff)
     Y[1] = Y[1] * Yi
     X = Xp + Kg.dot(Y - I.dot(Xp))
  */

  // Un-normalize the gradient using the measurement
  if (D == 2) {
    float Yd = Y[0] - Xp[0];                    // Compute difference between meas and predicted meas
    X[0] = Xp[0] + (Yd * Kg[0]) + (Yd * Kg[2]); // Compute new meas state

    // Un-normalize the grad
    float Y_sign = copysignf(1.0, Y[1]); // copy sign of the gradient, note: this will give +1 or -1 for floating point 0, not 0
    float Y_abs = fabsf(Y[1]);
    float X_abs = fabsf(X[0]);

    if (X_abs == 0.f) {
      if (X[-2] == 0.f) {
        // if current and previous meas state is zero, then gradient is not un-normalized
        Yd = Y[1] - Xp[1];
      }
      else {
        // Compute difference between grad and predicted grad using previous state
        X_abs = fabsf(X[-2]);
        Yd = (Y_sign * (Y_abs * X_abs)) - Xp[1];
      }
    }
    if (X_abs < THRES) {
      X_abs = THRES;
      Yd = (Y_sign * (Y_abs * X_abs)) - Xp[1];
    }
    else {
      // Compute difference between grad and predicted with denormalized process
      Yd = (Y_sign * (Y_abs * X_abs)) - Xp[1]; 
    }
    
    // Compute new grad state
    X[1] = Xp[1] + (Yd * Kg[1]) + (Yd * Kg[3]);
    return;
  }
  else {
    // print a warning if (D != 2)
    static bool not_warned = true;
    if (not_warned) {
      printf("WARNING: update state normed not implemented for %d dims!!!\n", D);
      not_warned = false;
    }
    // Y - Xp
    float* Y_Xp = (float*)malloc(sizeof(float) * D);
    matrix_subtraction(Y_Xp, Y, Xp, D);

    // Kg dot (Y - Xp)
    float* Kg_Y_Xp = (float*)malloc(sizeof(float) * D);
    v_dot_square(Kg_Y_Xp, Kg, Y_Xp, D);

    // Xp + Kg dot (Y - Xp)
    matrix_addition(X, Xp, Kg_Y_Xp, D);
    
    free(Y_Xp);
    free(Kg_Y_Xp);
  }
}

void update_covariance(float* P, const float* Pp, const float* H, const float* Kg, const float* I, const int D) {
  /* Update the state covariance matrix based on the measurement and predicted state covariances
     P = (I - Kg.dot(H)).dot(Pp)
  */

  // Kg dot H
  float* Kg_H = (float*)malloc(sizeof(float) * D * D);
  square_matrix_product(Kg_H, Kg, H, D);

  // I - (Kg dot H)
  float* I_Kg_H = (float*)malloc(sizeof(float) * D * D);
  matrix_subtraction(I_Kg_H, I, Kg_H, D * D);

  // (I - (Kg dot H)) dot Pp
  square_matrix_product(P, I_Kg_H, Pp, D);

  free(Kg_H);
  free(I_Kg_H);
}

void smooth_gain(float* Sg, const float* Pf, const float* Pr, const float* Hf, const float* Hr, const float* I, const int D) {
  /* Compute the smoothing gain matrix based on forward and reverse covariance matrices
     Sg = (Pf.dot(Hf) + Pr.dot(Hr) ).dot(np.linalg.pinv((H.dot(Pf).dot(I.T) + Pr)))
  */

  // Compute Pf_IT = Pf dot I.T
  float* Pf_IT = (float*)malloc(sizeof(float) * D*D);
  square_matrix_product_t(Pf_IT, Pf, I, D);

  // Compute I_Pf_IT = I dot Pf_IT
  float* I_Pf_IT = (float*)malloc(sizeof(float) * D*D);
  square_matrix_product(I_Pf_IT, I, Pf_IT, D);

  // compute I_Pf_IT_Pr = I_Pf_IT + Pr
  float* I_Pf_IT_Pr = (float*)malloc(sizeof(float) * D*D);
  matrix_addition(I_Pf_IT_Pr, I_Pf_IT, Pr, D*D);

  // Compute I_Pf_IT_Pr_INV = pinv(I_Pf_IT_Pr)
  float* I_Pf_IT_Pr_INV = (float*)malloc(sizeof(float) * D*D);
  inverse_matrix(I_Pf_IT_Pr_INV, I_Pf_IT_Pr, D);

  // Compute Pf_Hf = Pf dot Hf
  float* Pf_Hf = (float*)malloc(sizeof(float) * D*D);
  square_matrix_product(Pf_Hf, Pf, Hf, D);

  // Compute Pr_Hr = Pr dot Hr
  float* Pr_Hr = (float*)malloc(sizeof(float) * D*D);
  square_matrix_product(Pr_Hr, Pr, Hr, D);

  // Compute Pf_Hf_Pr_Hr = Pf_Hf + Pr_Hr
  float* Pf_Hf_Pr_Hr = (float*)malloc(sizeof(float) * D*D);
  matrix_addition(Pf_Hf_Pr_Hr, Pf_Hf, Pr_Hr, D*D);

  // Compute Sg = Pf_Hf_Pr_Hr dot I_Pf_IT_Pr_INV
  square_matrix_product(Sg, Pf_Hf_Pr_Hr, I_Pf_IT_Pr_INV, D);

  free(Pf_IT);
  free(I_Pf_IT);
  free(I_Pf_IT_Pr);
  free(I_Pf_IT_Pr_INV);
  free(Pf_Hf);
  free(Pr_Hr);
  free(Pf_Hf_Pr_Hr);
}

void multi_smooth_gain(float* Sg, const float* P, const float* H, const float* Psum_inv, const int N, const int D) {
  /* Compute the smoothing gain matrix based on various number of covariance matrices
     for i in N:
      Sg[i] = (P[i].dot(H[i])).dot(Psum_inv)
  */
  float* P_H = (float*)malloc(sizeof(float) * D * D);
  for (int i=0; i<N; i++) {
    int ni = i * D * D;
    square_matrix_product(P_H, &P[ni], &H[ni], D);
    square_matrix_product(&Sg[ni], P_H, Psum_inv, D);
  }
  free(P_H);
}

void smooth_covariance(float* Ps, const float* Pf, const float* Pr, const float* Sg, const float* I, const int D) {
  /* Compute the smoothed covariance matrix based on forward and reverse covariance matrices
     Ps = Pf + Sg dot (Pr - [I dot Pf])
  */

  // I_Pf = I dot Pf
  float* I_Pf = (float*)malloc(sizeof(float) * D*D);
  square_matrix_product(I_Pf, I, Pf, D);

  // Pr_I_Pf = Pr - I_Pf
  float* Pr_I_Pf = (float*)malloc(sizeof(float) * D*D);
  matrix_subtraction(Pr_I_Pf, Pr, I_Pf, D*D);

  // Sg_Pr_I_Pf = Sg dot Pr_I_Pf
  float* Sg_Pr_I_Pf = (float*)malloc(sizeof(float) * D*D);
  square_matrix_product(Sg_Pr_I_Pf, Sg, Pr_I_Pf, D);

  // Ps = Pf + Sg_Pr_I_Pf
  matrix_addition(Ps, Pf, Sg_Pr_I_Pf, D*D);

  free(I_Pf);
  free(Pr_I_Pf);
  free(Sg_Pr_I_Pf);
}

void kf_pass(float* X, float* Xp, float* P, float* Pp, const float* Y, const float* R, const float* H, const float Q, const int XN, const int N, const int D, const bool GNORM, const float T) {
  /* Compute a 1D Kalman pass across the provided XN to N measurement (Y) array
  */

  const float A[4] = {1.f,1.f,0.f,1.f}; // Transition matrix   // TODO: provide A matrix as arg

  float* I = (float*)malloc(sizeof(float) * (D*D));
  set_square_matrix_identity(I, D);

  float* Kg = (float*)calloc(D*D, sizeof(float));

  for (int i=XN+1; i < N; i++) {
    int xi = i * D;             // Current X index
    int xim = (i - 1) * D;      // Previous X index
    int pi = i * D * D;         // Current P index
    int pim = (i - 1) * D * D;  // Previous P index

    // Compute the new (predicted) state
    prediction_1st_order(&Xp[xi], &X[xim], A, D);

    // Compute the predicted process covariance matrix
    predicted_covariance(&Pp[pi], &P[pim], A, Q, D);
    
    // Multiply by identity matrix
    for (int i =0; i < D*D; i++) {
      Pp[pi+i] *= I[i];
    }

    // compute kalman gain
    kalman_gain(Kg, &Pp[pi], &R[pi], &H[pi], D);

    // calculate new state
    if (GNORM) {
      update_state_normed(&X[xi], &Xp[xi], &Y[xi], Kg, D, T);
    }
    else
      update_state(&X[xi], &Xp[xi], &Y[xi], Kg, D);

    // calculate new covariance matrix
    update_covariance(&P[pi], &Pp[pi], &H[pi], Kg, I, D);
  }

  free(I);
  free(Kg);
}

void smooth(float* Xs, float* Ps, const float* Xf, const float* Xr, const float* Pf, const float* Pr, const float* Hf, const float* Hr, const int N, const int D) {
  /* Compute a 1D smoothing pass across the provided N forward and reverse arrays
  */

  float* I = (float*)malloc(sizeof(float) * (D*D));
  set_square_matrix_identity(I, D);

  float* Sg = (float*)calloc(D*D, sizeof(float));

  for (int i=0; i<N; i++) {

    int xi = i * D;             // Current X index
    int pi = i * D * D;         // Current P index

    // smoothing gain
    // Sg = (Pf.dot(Hf) + Pr.dot(Hr) ).dot(np.linalg.pinv((H.dot(Pf).dot(I.T) + Pr)))
    smooth_gain(Sg, &Pf[pi], &Pr[pi], &Hf[pi], &Hr[pi], I, D);
    
    // compute smoothed state
    // Xs = Xf + Sg.dot(Xr - I.dot(Xf))
    update_state(&Xs[xi], &Xf[xi], &Xr[xi], Sg, D);

    // compute smoothed state covariance
    // Ps = Pf + Sg.dot(Pr - I.dot(Pf))
    smooth_covariance(&Ps[pi], &Pf[pi], &Pr[pi], Sg, I, D);
  }

  free(I);
  free(Sg);
}

void multi_smooth(float* Xs, float* Ps, const float* X, const float* P, const float* H, const int M, const int D) {
  /* Compute the smoothed cell of the provided M pass cells
  */

  float* P_inv = (float*)calloc(M*D*D, sizeof(float));
  float* Sg = (float*)calloc(M*D*D, sizeof(float));
  float* Psum = (float*)calloc(D*D, sizeof(float));
  float* Psum_inv = (float*)calloc(D*D, sizeof(float));

  // compute P_inv
  for (int i=0; i<M; i++) {
    inverse_matrix(&P_inv[i*D*D], &P[i*D*D], D);
  }

  // compute Psum
  // Psum = Pfx.dot(Hfx) + Prx.dot(Hrx) + Pfy.dot(Hfy) + Pry.dot(Hry)
  multi_sum_square_product(Psum, P_inv, H, M, D);

  // compute Psum_inv
  // Psum_inv = np.linalg.pinv(Psum)
  inverse_matrix(Psum_inv, Psum, D);

  // compute Sg
  multi_smooth_gain(Sg, P_inv, H, Psum_inv, M, D);

  // compute smooth state
  multi_sum_v_dot_square(Xs, Sg, X, M, D);

  // compute smoothed state covariance
  multi_sum_square_product(Ps, Sg, P, M, D);

  free(P_inv);
  free(Sg);
  free(Psum);
  free(Psum_inv);
}

void spatial_kf_pass(float* X, float* Xp, float* P, float* Pp, const float* Y, const float* R, const float* H, const int* XN, const float Q, const int NROWS, const int NCOLS, const int D, const bool GNORM, const float T, const int NUMT) {
  /* Compute the kalman pass across the columns of a 2D raster
     loop through the raster, providing each row to kf_pass
  */

  // Set the number of threads to use
  omp_set_num_threads(NUMT);

  int r;
  #pragma omp parallel for default(none) shared(X, Xp, P, Pp, Y, R, H, XN, Q, NROWS, NCOLS, D, GNORM, T)
  for (r=0; r<NROWS; r++) {
    // Only run pass if row contains valid measurements
    if (XN[r] < NCOLS) {
      int xi = r * NCOLS * D;
      int pi = r * NCOLS * D * D;
      kf_pass(&X[xi], &Xp[xi], &P[pi], &Pp[pi], &Y[xi], &R[pi], &H[pi], Q, XN[r], NCOLS, D, GNORM, T);
    }
  }
}

void spatial_smooth(float* Xs, float* Ps, const float* Xf, const float* Xr, const float* Pf, const float* Pr, const float* Hf, const float* Hr, const bool* vr, const int NROWS, const int NCOLS, const int D, const int NUMT) {
  /* Compute the smoothing filter across the columns of a 2D raster
     loop through the raster, providing each row to smooth
  */

  // Set the number of threads to use
  omp_set_num_threads(NUMT);

  int r;
  #pragma omp parallel for default(none) shared(Xs, Ps, Xf, Xr, Pf, Pr, Hf, Hr, vr, NROWS, NCOLS, D)
  for (r=0; r<NROWS; r++) {
    // Only run smooth for COLS containing valid inputs 
    if (vr[r]) {
      int xi = r * NCOLS * D;
      int pi = r * NCOLS * D * D;
      smooth(&Xs[xi], &Ps[pi], &Xf[xi], &Xr[xi], &Pf[pi], &Pr[pi], &Hf[pi], &Hr[pi], NCOLS, D);
    }
  }
}

void spatial_multi_smooth(float* Xs, float* Ps, const float* X, const float* P, const float* H, const int* idx, const int N, const int M, const int D, const int NUMT) {
  /* Compute the smoothing filter across a 2D raster
     loop through the raster, providing passes for each cell to multi_smooth
  */

  // Set the number of threads to use
  omp_set_num_threads(NUMT);

  int i;
  #pragma omp parallel for default(none) shared(Xs, Ps, X, P, H, idx, N, M, D)
  for (i=0; i<N; i++) {
    int ni = idx[i];
    int xi = ni * D;
    int xim = xi * M;
    int pi = ni * D * D;
    int pim = pi * M;
    multi_smooth(&Xs[xi], &Ps[pi], &X[xim], &P[pim], &H[pim], M, D);
  }
}
