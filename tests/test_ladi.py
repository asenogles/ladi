import unittest
import sys
import numpy as np
sys.path.append('.')
import ladi.ladi as ladi

np.random.seed(1)

class Test_ladi(unittest.TestCase):
    def test_prediction_1st_order(self):
        X = np.random.uniform(0, 100, size=(2))
        A = np.array([[1,1],[0,1]], dtype=np.float32)
        ans1 = ladi.prediction_1st_order(X, A)
        ans2 = np.dot(A.astype(np.float32), X.astype(np.float32)).astype(np.float32)
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=5)

    def test_predicted_covariance(self):
        P = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        A = np.array([[1,1],[0,1]], dtype=np.float32)
        Q = 0.05
        ans1 = ladi.predicted_covariance(P, A, Q=Q)
        ans2 = np.dot(A, np.dot(P, A.T)) + Q
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=5)
    
    def test_kalman_gain(self):
        P = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        R = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        H = np.identity(2, dtype=np.float32)
        ans1 = ladi.kalman_gain(P, R, H)
        ans2 = P.dot(H.T).dot(np.linalg.pinv((H.dot(P).dot(H.T) + R)))
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=5)

    def test_update_state(self):
        Xp = np.random.uniform(0, 100, size=(2)).astype(np.float32)
        Y = np.random.uniform(0, 100, size=(2)).astype(np.float32)
        Kg = np.random.uniform(0, 1, size=(2,2)).astype(np.float32)
        I = np.identity(2, dtype=np.float32)
        ans1 = ladi.update_state(Xp, Y, Kg)
        ans2 = Xp + np.dot(Kg, Y - np.dot(I, Xp))
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=5)

    def test_update_covariance(self):
        Pp = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        H = np.array([[1, 0], [0, 1]], dtype=np.float32)
        Kg = np.random.uniform(0, 1, size=(2,2)).astype(np.float32)
        I = np.identity(2, dtype=np.float32)
        ans1 = ladi.update_covariance(Pp, H, Kg)
        ans2 = np.dot(I - np.dot(Kg, H), Pp)
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=5)

    def test_smooth_gain(self):
        Pf = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        Pr = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        Hf = np.identity(2, dtype=np.float32)
        Hr = np.identity(2, dtype=np.float32)
        I = np.identity(2, dtype=np.float32)
        ans1 = ladi.smooth_gain(Pf, Pr, Hf, Hr)
        ans2 = (Pf.dot(Hf) + Pr.dot(Hr) ).dot(np.linalg.pinv((I.dot(Pf).dot(I.T) + Pr)))
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=5)

    def test_multi_smooth_gain(self):
        D = 2
        N = 4
        P = np.random.uniform(0, 100, size=(N,D,D)).astype(np.float32)
        H = np.tile(np.eye(D), (N, 1, 1)).astype(np.float32)
        Psum_inv = np.linalg.pinv(np.linalg.pinv(P).sum(axis=0)).astype(np.float32)
        ans1 = ladi.multi_smooth_gain(P, H)
        ans2 = np.zeros((N,D,D), dtype=np.float32)
        for i in range(N):
            ans2[i] = (P[i].dot(H[i])).dot(Psum_inv)
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=3)

    def test_smooth_covariance(self):
        Pf = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        Pr = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        Sg = np.random.uniform(0, 1, size=(2,2)).astype(np.float32)
        I = np.identity(2, dtype=np.float32)
        ans1 = ladi.smooth_covariance(Pf, Pr, Sg)
        ans2 = Pf + Sg.dot(Pr - I.dot(Pf))
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=5)

if __name__ == '__main__':
    unittest.main()