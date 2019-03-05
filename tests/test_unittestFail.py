import unittest
class TestFailUnittest(unittest.TestCase):
    def test_fail(self):
        self.assertTrue(False)

suite1 = unittest.TestLoader().loadTestsFromTestCase(TestFailUnittest)
suite = unittest.TestSuite([suite1])

if __name__ in  ['__main__','builtins']:
    print("Running unittest test_unittestFail")
    result = unittest.TextTestRunner().run(suite)