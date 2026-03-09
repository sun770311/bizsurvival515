import unittest

class TestDummy(unittest.TestCase):
    def test_basic_math(self):
        """A simple dummy test to ensure the test suite runs."""
        self.assertEqual(1 + 1, 2)

if __name__ == '__main__':
    unittest.main()