try:
    import unittest
    import numpy as np
    import time
    import math

    from PyCo.Topography.ScalarParameters import rms_curvature, rms_slope, rms_height

except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)


class SinewaveTest(unittest.TestCase):
    def setUp(self):
        n = 256
        X, Y = np.mgrid[slice(0,n),slice(0,n)]

        self.hm = 0.1
        self.L = n
        self.sinsurf = np.sin(2 * np.pi / self.L * X) * np.sin(2 * np.pi / self.L * Y) * self.hm
        self.size= (self.L,self.L)

        self.precision = 5

    def test_rms_curvature(self):
        numerical = rms_curvature(self.sinsurf, size=self.size)
        analytical = np.sqrt(16*np.pi**4 *self.hm**2 / self.L**4 )
        #print(numerical-analytical)
        self.assertAlmostEqual(numerical,analytical,self.precision)

    def test_rms_slope(self):
        numerical = rms_slope(self.sinsurf, size=self.size)
        analytical = np.sqrt(2*np.pi ** 2 * self.hm**2 / self.L**2)
        # print(numerical-analytical)
        self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_height(self):
        numerical = rms_height(self.sinsurf )
        analytical = np.sqrt(self.hm**2 / 4)

        self.assertEqual(numerical,analytical)


if __name__ == '__main__':
    unittest.main()
