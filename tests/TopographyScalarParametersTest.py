try:
    import unittest
    import numpy as np
    import time
    import math

    import PyCo.Topography.Uniform.ScalarParameters as Uniform
    import PyCo.Topography.Nonuniform.ScalarParameters as Nonuniform

except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)


class SinewaveTestUniform(unittest.TestCase):
    def setUp(self):
        n = 256
        X, Y = np.mgrid[slice(0,n),slice(0,n)]

        self.hm = 0.1
        self.L = n
        self.sinsurf = np.sin(2 * np.pi / self.L * X) * np.sin(2 * np.pi / self.L * Y) * self.hm
        self.size= (self.L,self.L)

        self.precision = 5

    def test_rms_curvature(self):
        numerical = Uniform.rms_curvature(self.sinsurf, size=self.size)
        analytical = np.sqrt(16*np.pi**4 *self.hm**2 / self.L**4 )
        #print(numerical-analytical)
        self.assertAlmostEqual(numerical,analytical,self.precision)

    def test_rms_slope(self):
        numerical = Uniform.rms_slope(self.sinsurf, size=self.size)
        analytical = np.sqrt(2*np.pi ** 2 * self.hm**2 / self.L**2)
        # print(numerical-analytical)
        self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_height(self):
        numerical = Uniform.rms_height(self.sinsurf )
        analytical = np.sqrt(self.hm**2 / 4)

        self.assertEqual(numerical,analytical)


class SinewaveTestNonuniform(unittest.TestCase):
    def setUp(self):
        n = 256

        self.hm = 0.1
        self.L = n
        self.X = np.arange(n+1)  # n+1 because we need the endpoint
        self.sinsurf = np.sin(2 * np.pi * self.X / self.L) * self.hm

        self.precision = 5

#    def test_rms_curvature(self):
#        numerical = Nonuniform.rms_curvature(self.X, self.sinsurf)
#        analytical = np.sqrt(16*np.pi**4 *self.hm**2 / self.L**4 )
#        #print(numerical-analytical)
#        self.assertAlmostEqual(numerical,analytical,self.precision)

    def test_rms_slope(self):
        numerical = Nonuniform.rms_slope(self.X, self.sinsurf)
        analytical = np.sqrt(2*np.pi ** 2 * self.hm**2 / self.L**2)
        # print(numerical-analytical)
        self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_height(self):
        numerical = Nonuniform.rms_height(self.X, self.sinsurf)
        analytical = np.sqrt(self.hm**2 / 2)
        #numerical = np.sqrt(np.trapz(self.sinsurf**2, self.X))

        self.assertAlmostEqual(numerical, analytical, self.precision)

if __name__ == '__main__':
    unittest.main()
