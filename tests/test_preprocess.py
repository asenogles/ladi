import unittest
import sys
import numpy as np
sys.path.append('.')
import ladi.preprocess as pp

np.random.seed(1)

class Test_preprocess(unittest.TestCase):
    def test_round_to_zero(self):
        T = 0.2
        arr = np.random.normal(0., 1., size=(64,64))
        rounded_arr = pp.round_to_zero(arr, T)
        sum1 = (np.absolute(arr) < T).sum()
        sum2 = (rounded_arr == 0.).sum()
        self.assertEqual(sum1, sum2)



if __name__ == '__main__':
    unittest.main()