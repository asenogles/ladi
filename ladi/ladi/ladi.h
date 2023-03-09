#pragma once

void prediction_1st_order(float* Xp, const float* Xi, const float* A, const int D);

void predicted_covariance(float* Pp, const float* Pi, const float* A, const float Q, const int D);

void kalman_gain(float* Kg, const float* P, const float* R, const float* H, const int D);

void update_state(float* X, const float* Xp, const float* Y, const float* Kg, const int D);

void update_state_normed(float* X, const float* Xp, const float* Y, const float* Kg, const int D, const float THRES);

void update_covariance(float* P, const float* Pp, const float* H, const float* Kg, const float* I, const int D);

void smooth_gain(float* Sg, const float* Pf, const float* Pr, const float* Hf, const float* Hr, const float* I, const int D);

void multi_smooth_gain(float* Sg, const float* P, const float* H, const float* Psum_inv, const int N, const int D);

void smooth_covariance(float* Ps, const float* Pf, const float* Pr, const float* Sg, const float* I, const int D);

void kf_pass(float* X, float* Xp, float* P, float* Pp, const float* Y, const float* R, const float* H, const float Q, const int XN, const int N, const int D, const bool GNORM, const float T);

void smooth(float* Xs, float* Ps, const float* Xf, const float* Xr, const float* Pf, const float* Pr, const float* Hf, const float* Hr, const int N, const int D);

void multi_smooth(float* Xs, float* Ps, const float* X, const float* P, const float* H, const int M, const int D);

void spatial_kf_pass(float* X, float* Xp, float* P, float* Pp, const float* Y, const float* R, const float* H, const int* XN, const float Q, const int NROWS, const int NCOLS, const int D, const bool GNORM, const float T, const int NUMT);

void spatial_smooth(float* Xs, float* Ps, const float* Xf, const float* Xr, const float* Pf, const float* Pr, const float* Hf, const float* Hr, const bool* vr, const int NROWS, const int NCOLS, const int D, const int NUMT);

void spatial_multi_smooth(float* Xs, float* Ps, const float* X, const float* P, const float* H, const int* idx, const int N, const int M, const int D, const int NUMT);