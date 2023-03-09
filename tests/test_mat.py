import unittest
import sys
import numpy as np
sys.path.append('.')
import ladi.mat as mat

np.random.seed(1)

class Test_Mat(unittest.TestCase):
    def test_v_dot_square(self):
        X = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        Y = np.random.uniform(0, 100, size=(2)).astype(np.float32)
        ans1 = mat.v_dot_square(X, Y)
        ans2 = np.dot(X, Y).astype(np.float32)
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=4)

    def test_square_matrix_product(self):
        X = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        Y = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        ans1 = mat.square_matrix_product(X, Y)
        ans2 = np.dot(X, Y).astype(np.float32)
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=4)

    def test_square_matrix_product_t(self):
        X = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        Y = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        ans1 = mat.square_matrix_product_t(X, Y)
        ans2 = np.dot(X, Y.T).astype(np.float32)
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=4)

    def test_t_square_matrix_product(self):
        X = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        Y = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        ans1 = mat.t_square_matrix_product(X, Y)
        ans2 = np.dot(X.T, Y).astype(np.float32)
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=3)

    def test_determinant(self):
        X = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        ans1 = mat.determinant(X)
        ans2 = np.linalg.det(X).astype(np.float32)
        self.assertAlmostEqual(ans1, ans2, places=3)

    def test_inverse(self):
        X = np.random.uniform(0, 100, size=(2,2)).astype(np.float32)
        ans1 = mat.inverse(X)
        ans2 = np.linalg.pinv(X).astype(np.float32)
        np.testing.assert_array_almost_equal(ans1, ans2, decimal=4)

if __name__ == '__main__':
    unittest.main()