# distutils: language = c
# distutils: sources = ladi/ladi.c, mat/mat.c, pinv/pinv.c

import numpy as np
cimport cython
cimport numpy as np

from libcpp cimport bool

##############################################################
####################### PINV Operations ######################
##############################################################
cdef extern from "pinv/pinv.h":
    void call_pinv(float* pinv_mat, const float* mat, const int NROWS, const int NCOLS)

def cy_pinv(const float [:,::1] mat):
    
    cdef int NROWS = mat.shape[0]
    cdef int NCOLS = mat.shape[1]

    # create pseudo inverse array
    pinv_mat = np.empty((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] pinv_mat_ptr = pinv_mat
    call_pinv(&pinv_mat_ptr[0,0], &mat[0,0], NROWS, NCOLS)
    return pinv_mat

##############################################################
####################### MAT Operations #######################
##############################################################
cdef extern from "mat/mat.h":
    void v_dot_square(float* A, const float* X, const float* Y, const int D)
    void multi_sum_v_dot_square(float* A, const float* X, const float* Y, const int N, const int D)
    void square_matrix_product(float* A, const float* X, const float* Y, const int D)
    void square_matrix_product_t(float* A, const float* X, const float* Y, const int D)
    void t_square_matrix_product(float* A, const float* X, const float* Y, const int D)
    void multi_sum_square_product(float* A, const float* X, const float* Y, const int N, const int D)
    void determinant(const float* mat, float* det, const int D)
    bool inverse_matrix(float* inv, const float* mat, const int D)

def cy_v_dot_square(const float[:,::1] X, const float[::1] Y):
    assert(Y.shape[0] == X.shape[0])
    assert(X.shape[0] == X.shape[1])
    cdef int d = X.shape[0]

    # create output array
    A = np.zeros((d), dtype=np.float32, order='C')
    cdef float[::1] A_ptr = A
    v_dot_square(&A_ptr[0], &X[0,0], &Y[0], d)
    return A

def cy_multi_sum_v_dot_square(const float[:,:,::1] X, const float[:,::1] Y):
    assert(Y.shape[0] == X.shape[0])
    assert(Y.shape[1] == X.shape[1])
    cdef int N = X.shape[0]
    cdef int NROWS = X.shape[1]
    cdef int NCOLS = X.shape[2]
    assert(NROWS == NCOLS)

    # create output array
    A = np.zeros((NROWS), dtype=np.float32, order='C')
    cdef float[::1] A_ptr = A
    multi_sum_v_dot_square(&A_ptr[0], &X[0,0,0], &Y[0,0], N, NROWS)
    return A

def cy_square_matrix_product(const float[:,::1] X, const float[:,::1] Y):
    assert(Y.shape[0] == X.shape[0])
    assert(Y.shape[1] == X.shape[1])
    cdef int NROWS = X.shape[0]
    cdef int NCOLS = X.shape[1]
    assert(NROWS == NCOLS)

    # create predicted array
    A = np.empty((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] A_ptr = A
    square_matrix_product(&A_ptr[0,0], &X[0,0], &Y[0,0], NROWS)
    return A

def cy_square_matrix_product_t(const float[:,::1] X, const float[:,::1] Y):
    assert(Y.shape[0] == X.shape[0])
    assert(Y.shape[1] == X.shape[1])
    cdef int NROWS = X.shape[0]
    cdef int NCOLS = X.shape[1]
    assert(NROWS == NCOLS)

    # create predicted array
    A = np.empty((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] A_ptr = A
    square_matrix_product_t(&A_ptr[0,0], &X[0,0], &Y[0,0], NROWS)
    return A

def cy_t_square_matrix_product(const float[:,::1] X, const float[:,::1] Y):
    assert(Y.shape[0] == X.shape[0])
    assert(Y.shape[1] == X.shape[1])
    cdef int NROWS = X.shape[0]
    cdef int NCOLS = X.shape[1]
    assert(NROWS == NCOLS)

    # create predicted array
    A = np.empty((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] A_ptr = A
    t_square_matrix_product(&A_ptr[0,0], &X[0,0], &Y[0,0], NROWS)
    return A

def cy_multi_sum_square_product(const float[:,:,::1] X, const float[:,:,::1] Y):
    assert(X.shape[0] == Y.shape[0])
    assert(X.shape[1] == Y.shape[1])
    assert(X.shape[2] == Y.shape[2])
    
    cdef int N = X.shape[0]
    cdef int NROWS = X.shape[1]
    cdef int NCOLS = X.shape[2]
    assert(NROWS == NCOLS)

    # create output array
    A = np.zeros((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] A_ptr = A

    multi_sum_square_product(&A_ptr[0,0], &X[0,0,0], &Y[0,0,0], N, NROWS)
    return A

def cy_determinant(const float[:,::1] mat):
    cdef int NROWS = mat.shape[0]
    cdef int NCOLS = mat.shape[1]
    assert(NROWS == NCOLS)
    cdef float D = 0.
    cdef float* D_ptr = &D
    determinant(&mat[0,0], D_ptr, NROWS)
    return D

def cy_inverse_matrix(const float[:,::1] mat):
    cdef int NROWS = mat.shape[0]
    cdef int NCOLS = mat.shape[1]
    assert(NROWS == NCOLS)

    # create predicted array
    inv = np.empty((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] inv_ptr = inv

    inverse_matrix(&inv_ptr[0,0], &mat[0,0], NROWS)
    return inv


##############################################################
####################### LADI Operations ######################
##############################################################
cdef extern from "ladi/ladi.h":
    void prediction_1st_order(float* Xp, const float* Xi, const float* A, const int D)
    void predicted_covariance(float* Pp, const float* Pi, const float* A, const float Q, const int D)
    void kalman_gain(float* Kg, const float* P, const float* R, const float* H, const int D)
    void update_state(float* X, const float* Xp, const float* Kg, const float* Y, const int D)
    void update_covariance(float* P, const float* Pp, const float* H, const float* Kg, const float* I, const int D)
    void smooth_gain(float* Sg, const float* Pf, const float* Pr, const float* Hf, const float* Hr, const float* I, const int D)
    void multi_smooth_gain(float* Sg, const float* P, const float* H, const float* Psum_inv, const int N, const int D)
    void smooth_covariance(float* Ps, const float* Pf, const float* Pr, const float* Sg, const float* I, const int D)
    void kf_pass(float* X, float* Xp, float* P, float* Pp, const float* Y, const float* R, const float* H, const float Q, const int XN, const int N, const int D, const bool GNORM, const float T)
    void smooth(float* Xs, float* Ps, const float* Xf, const float* Xr, const float* Pf, const float* Pr, const float* Hf, const float* Hr, const int N, const int D)
    void multi_smooth(float* Xs, float* Ps, const float* X, const float* P, const float* H, const int M, const int D)
    void spatial_kf_pass(float* X, float* Xp, float* P, float* Pp, const float* Y, const float* R, const float* H, const int* XN, const float Q, const int NROWS, const int NCOLS, const int D, const bool GNORM, const float T, const int NUMT)
    void spatial_smooth(float* Xs, float* Ps, const float* Xf, const float* Xr, const float* Pf, const float* Pr, const float* Hf, const float* Hr, const bool* vr, const int NROWS, const int NCOLS, const int D, const int NUMT)
    void spatial_multi_smooth(float* Xs, float* Ps, const float* X, const float* P, const float* H, const int* idx, const int N, const int M, const int D, const int NUMT)

def cy_prediction_1st_order(const float[::1] Xi, const float[:,::1] A):
    assert(Xi.shape[0] == A.shape[0] == A.shape[1])
    cdef int D = Xi.shape[0]

    # create predicted array
    Xp = np.zeros(D, dtype=np.float32, order='C')
    cdef float[::1] Xp_ptr = Xp
    prediction_1st_order(&Xp_ptr[0], &Xi[0], &A[0,0], D)
    return Xp

def cy_predicted_covariance(const float[:,::1] Pi, const float[:,::1] A, const float Q):
    assert(Pi.shape[0] == A.shape[0] == Pi.shape[1] == A.shape[1])
    cdef int NROWS = Pi.shape[0]
    cdef int NCOLS = Pi.shape[1]

    # create predicted array
    Pp = np.empty((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] Pp_ptr = Pp
    predicted_covariance(&Pp_ptr[0,0], &Pi[0,0], &A[0,0], Q, NROWS)
    return Pp

def cy_kalman_gain(const float[:,::1] P, const float[:,::1] R, const float[:,::1] H):
    assert(P.shape[0] == R.shape[0] == H.shape[0])
    assert(P.shape[1] == R.shape[1] == H.shape[1])
    cdef int NROWS = P.shape[0]
    cdef int NCOLS = P.shape[1]
    assert(NROWS == NCOLS)

    # create kalman gain array
    Kg = np.empty((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] Kg_ptr = Kg
    kalman_gain(&Kg_ptr[0,0], &P[0,0], &R[0,0], &H[0,0], NROWS)
    return Kg

def cy_update_state(const float[::1] Xp, const float[::1] Y, const float[:,::1] Kg):
    assert(Xp.shape[0] == Y.shape[0] == Kg.shape[0] == Kg.shape[1])
    cdef int D = Kg.shape[0]

    # create new state array
    X = np.empty(D, dtype=np.float32, order='C')
    cdef float[::1] X_ptr = X
    update_state(&X_ptr[0], &Xp[0], &Y[0], &Kg[0,0], D)
    return X

def cy_update_covariance(const float[:,::1] Pp, const float[:,::1] H, const float [:,::1] Kg):
    assert(Pp.shape[0] == H.shape[0] == Kg.shape[0])
    assert(Pp.shape[1] == H.shape[1] == Kg.shape[1])
    cdef int NROWS = Kg.shape[0]
    cdef int NCOLS = Kg.shape[1]
    assert(NROWS == NCOLS)

    # create new covariance array
    P = np.empty((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] P_ptr = P

    # create eye matrix
    I = np.eye(NROWS, NCOLS, dtype=np.float32, order='C')
    cdef float[:,::1] I_ptr = I
    update_covariance(&P_ptr[0,0], &Pp[0,0], &H[0,0], &Kg[0,0], &I_ptr[0,0], NROWS)
    return P

def cy_smooth_gain(const float[:,::1] Pf, const float[:,::1] Pr, const float[:,::1] Hf, const float[:,::1] Hr):
    assert(Pf.shape[0] == Pr.shape[0] == Hf.shape[0] == Hr.shape[0])
    assert(Pf.shape[1] == Pr.shape[1] == Hf.shape[1] == Hr.shape[1])

    cdef int NROWS = Pf.shape[0]
    cdef int NCOLS = Pf.shape[1]
    assert(NCOLS == NROWS)

    # create kalman gain array
    Sg = np.empty((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] Sg_ptr = Sg

    I = np.eye(NROWS, NCOLS, dtype=np.float32, order='C')
    cdef float[:,::1] I_ptr = I

    smooth_gain(&Sg_ptr[0,0], &Pf[0,0], &Pr[0,0], &Hf[0,0], &Hr[0,0], &I_ptr[0,0], NROWS)
    return Sg

def cy_multi_smooth_gain(const float[:,:,::1] P, const float[:,:,::1] H, const float[:,::1] Psum_inv):
    assert(P.shape[0] == H.shape[0])
    assert(P.shape[1] == H.shape[1] == Psum_inv.shape[0])
    assert(P.shape[2] == H.shape[2] == Psum_inv.shape[1])

    # get dims
    cdef int N = P.shape[0]
    cdef int NROWS = P.shape[1]
    cdef int NCOLS = P.shape[2]
    assert(NROWS == NCOLS)

    # create Smoothing gain array
    Sg = np.zeros((N, NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,:,::1] Sg_ptr = Sg

    multi_smooth_gain(&Sg_ptr[0,0,0], &P[0,0,0], &H[0,0,0], &Psum_inv[0,0], N, NROWS)
    return Sg


def cy_smooth_covariance(const float[:,::1] Pf, const float[:,::1] Pr, const float [:,::1] Sg):
    assert(Pf.shape[0] == Pr.shape[0] == Sg.shape[0])
    assert(Pf.shape[1] == Pr.shape[1] == Sg.shape[1])

    cdef int NROWS = Pf.shape[0]
    cdef int NCOLS = Pf.shape[1]
    assert(NROWS == NCOLS)

    # create new covariance array
    Ps = np.empty((NROWS, NCOLS), dtype=np.float32, order='C')
    cdef float[:,::1] Ps_ptr = Ps

    # create eye matrix
    I = np.eye(NROWS, NCOLS, dtype=np.float32, order='C')
    cdef float[:,::1] I_ptr = I

    smooth_covariance(&Ps_ptr[0,0], &Pf[0,0], &Pr[0,0], &Sg[0,0], &I_ptr[0,0], NROWS)
    return Ps

def cy_kf_pass(const float[:,::1] Y, const float [:,:,::1] R, const float[:,:,::1] H, const float Q, const int XN, gnorm=False, T=0.01):

    # check dimensions
    assert(Y.shape[0] == R.shape[0] == H.shape[0])
    assert(Y.shape[1] == R.shape[1] == H.shape[1])
    assert(Y.shape[1] == R.shape[2] == H.shape[2])

    # get length & dims
    cdef int N = Y.shape[0]
    cdef int D = Y.shape[1]

    # create arrays
    X = np.zeros((N, D), dtype=np.float32, order='C')
    cdef float[:,::1] X_ptr = X
    Xp = np.zeros((N, D), dtype=np.float32, order='C')
    cdef float[:,::1] Xp_ptr = Xp
    P = np.zeros((N, D, D), dtype=np.float32, order='C')
    cdef float[:,:,::1] P_ptr = P
    Pp = np.zeros((N, D, D), dtype=np.float32, order='C')
    cdef float[:,:,::1] Pp_ptr = Pp

    # initialize first elements
    X[XN] = Y[XN]
    Xp[XN] = Y[XN]
    P[XN,:,:] = R[XN,:,:]
    Pp[XN,:,:] = R[XN,:,:]

    kf_pass(&X_ptr[0,0], &Xp_ptr[0,0], &P_ptr[0,0,0], &Pp_ptr[0,0,0], &Y[0,0], &R[0,0,0], &H[0,0,0], Q, XN, N, D, gnorm, T)

    return X, P, Xp, Pp

def cy_smooth(const float [:,::1] Xf, const float [:,::1] Xr, const float[:,:,::1] Pf, const float[:,:,::1] Pr, const float[:,:,::1] Hf, const float[:,:,::1] Hr):
    assert(Xf.shape[0] == Xr.shape[0] == Pf.shape[0] == Pr.shape[0] == Hf.shape[0] == Hr.shape[0])
    assert(Xf.shape[1] == Xr.shape[1] == Pf.shape[1] == Pr.shape[1] == Hf.shape[1] == Hr.shape[1])
    assert(Xf.shape[1] == Pf.shape[2] == Pr.shape[2] == Hf.shape[2] == Hr.shape[2])

    # get length & dims
    cdef int N = Xf.shape[0]
    cdef int D = Xf.shape[1]

   # create arrays
    Xs = np.zeros((N, D), dtype=np.float32, order='C')
    cdef float[:,::1] Xs_ptr = Xs
    Ps = np.zeros((N, D, D), dtype=np.float32, order='C')
    cdef float[:,:,::1] Ps_ptr = Ps

    smooth(&Xs_ptr[0,0], &Ps_ptr[0,0,0], &Xf[0,0], &Xr[0,0], &Pf[0,0,0], &Pr[0,0,0], &Hf[0,0,0], &Hr[0,0,0], N, D)
    return Xs, Ps

def cy_multi_smooth(const float[:,::1] X, const float[:,:,::1] P, const float[:,:,::1] H):
    assert(X.shape[0] == P.shape[0] == H.shape[0])
    assert(X.shape[1] == P.shape[1] == H.shape[1])
    assert(X.shape[1] == P.shape[2] == H.shape[2])

    # get length & dims
    cdef int M = X.shape[0]
    cdef int D = X.shape[1]

   # create arrays
    Xs = np.zeros((D), dtype=np.float32, order='C')
    cdef float[::1] Xs_ptr = Xs
    Ps = np.zeros((D, D), dtype=np.float32, order='C')
    cdef float[:,::1] Ps_ptr = Ps

    multi_smooth(&Xs_ptr[0], &Ps_ptr[0,0], &X[0,0], &P[0,0,0], &H[0,0,0], M, D)
    return Xs, Ps

def cy_spatial_kf_pass(const float[:,:,::1] Y, const float[:,:,:,::1] R, const float[:,:,:,::1] H, const int[::1] XN, const float Q, gnorm=False, T=0.01, numt=1):
    assert(Y.shape[0] == R.shape[0] == H.shape[0])
    assert(Y.shape[1] == R.shape[1] == H.shape[1])
    assert(Y.shape[2] == R.shape[2] == H.shape[2])
    assert(Y.shape[2] == R.shape[3] == H.shape[3])

    # get length & dims
    cdef int NROWS = Y.shape[0]
    cdef int NCOLS = Y.shape[1]
    cdef int D = Y.shape[2]

    # create arrays
    X = np.zeros((NROWS, NCOLS, D), dtype=np.float32, order='C')
    cdef float[:,:,::1] X_ptr = X
    Xp = np.zeros((NROWS, NCOLS, D), dtype=np.float32, order='C')
    cdef float[:,:,::1] Xp_ptr = Xp
    P = np.zeros((NROWS, NCOLS, D, D), dtype=np.float32, order='C')
    cdef float[:,:,:,::1] P_ptr = P
    Pp = np.zeros((NROWS, NCOLS, D, D), dtype=np.float32, order='C')
    cdef float[:,:,:,::1] Pp_ptr = Pp

    # initialize first elements
    for i, xn in enumerate(XN):
        if xn < NCOLS:
            X[i,xn,:] = Y[i,xn,:]
            Xp[i,xn,:] = Y[i,xn,:]
            P[i,xn,:,:] = R[i,xn,:,:]
            Pp[i,xn,:,:] = R[i,xn,:,:]
            # un-normalize the first measurement
            if gnorm:
                X[i,xn,1] = np.sign(X[i,xn,1]) * (np.abs(X[i,xn,1]) * np.abs(X[i,xn,0]))
                Xp[i,xn,1] = np.sign(Xp[i,xn,1]) * (np.abs(Xp[i,xn,1]) * np.abs(Xp[i,xn,0]))

    spatial_kf_pass(&X_ptr[0,0,0], &Xp_ptr[0,0,0], &P_ptr[0,0,0,0], &Pp_ptr[0,0,0,0], &Y[0,0,0], &R[0,0,0,0], &H[0,0,0,0], &XN[0], Q, NROWS, NCOLS, D, gnorm, T, numt)
    return X, Xp, P, Pp

def cy_spatial_smooth(const float[:,:,::1] Xf, const float[:,:,::1] Xr, const float[:,:,:,::1] Pf, const float[:,:,:,::1] Pr, const float[:,:,:,::1] Hf, const float[:,:,:,::1] Hr, const bool [::1] vr, numt=1):
    assert(Xf.shape[0] == Xr.shape[0] == Pf.shape[0] == Pr.shape[0] == Hf.shape[0] == Hr.shape[0] == vr.shape[0])
    assert(Xf.shape[1] == Xr.shape[1] == Pf.shape[1] == Pr.shape[1] == Hf.shape[1] == Hr.shape[1])
    assert(Xf.shape[2] == Xr.shape[2] == Pf.shape[2] == Pr.shape[2] == Hf.shape[2] == Hr.shape[2])
    assert(Xf.shape[2] == Xr.shape[2] == Pf.shape[3] == Pr.shape[3] == Hf.shape[3] == Hr.shape[3])

    # get length & dims
    cdef int NROWS = Xf.shape[0]
    cdef int NCOLS = Xf.shape[1]
    cdef int D = Xf.shape[2]

    # create arrays
    Xs = np.zeros((NROWS, NCOLS, D), dtype=np.float32, order='C')
    cdef float[:,:,::1] Xs_ptr = Xs
    Ps = np.zeros((NROWS, NCOLS, D, D), dtype=np.float32, order='C')
    cdef float[:,:,:,::1] Ps_ptr = Ps

    spatial_smooth(&Xs_ptr[0,0,0], &Ps_ptr[0,0,0,0], &Xf[0,0,0], &Xr[0,0,0], &Pf[0,0,0,0], &Pr[0,0,0,0], &Hf[0,0,0,0], &Hr[0,0,0,0], &vr[0], NROWS, NCOLS, D, numt)
    return Xs, Ps

def cy_spatial_multi_smooth(const float[:,:,:,::1] X, const float[:,:,:,:,::1] P, const float[:,:,:,:,::1] H, const int[::1] idx, numt=1):
    assert(X.shape[0] == P.shape[0] == H.shape[0])
    assert(X.shape[1] == P.shape[1] == H.shape[1])
    assert(X.shape[2] == P.shape[2] == H.shape[2])
    assert(X.shape[3] == P.shape[3] == H.shape[3])
    assert(X.shape[3] == P.shape[4] == H.shape[4])

    # get length & dims
    cdef int NROWS = X.shape[0]
    cdef int NCOLS = X.shape[1]
    cdef int N = idx.shape[0]
    cdef int M = X.shape[2]
    cdef int D = X.shape[3]

    # create arrays
    Xs = np.zeros((NROWS, NCOLS, D), dtype=np.float32, order='C')
    cdef float[:,:,::1] Xs_ptr = Xs
    Ps = np.zeros((NROWS, NCOLS, D, D), dtype=np.float32, order='C')
    cdef float[:,:,:,::1] Ps_ptr = Ps

    spatial_multi_smooth(&Xs_ptr[0,0,0], &Ps_ptr[0,0,0,0], &X[0,0,0,0], &P[0,0,0,0,0], &H[0,0,0,0,0], &idx[0], N, M, D, numt)
    return Xs, Ps

