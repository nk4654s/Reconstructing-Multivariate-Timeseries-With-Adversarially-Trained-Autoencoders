"""S3Prep/data/header.py"""

import unittest
from S3Prep.data import header

class TestHeader(unittest.TestCase):
    def setUp(self):
        self.header1 = header.Header()
        self.header2 = header.Header()

    def test_relative_complement_eqality(self):
        """Test Header.relative_complement method with equal headers"""
        self.header1.values = ['a', 'b', 'c']
        self.header2.values = ['a', 'b', 'c']

        self.header1.relative_complement(self.header2)
        self.assertEqual(self.header1.drop, [])

    def test_relative_complement_assertion_error(self):
        """Test Header.relative_complement method"""
        self.header1.values = ['a', 'b', 'c']
        self.header2.values = ['a', 'c', 'd']

        test = lambda: self.header1.relative_complement(self.header2)

        self.assertRaises(AssertionError, test)

    def test_relative_complement_drop_values(self):
        """Test Header.relative_complement method with drop values as return"""
        self.header1.values = ['a', 'b', 'c']
        self.header2.values = ['a', 'c']

        self.header1.relative_complement(self.header2)

        self.assertEqual(len(self.header1.drop), 1)
        self.assertTrue('b' in self.header1.drop)
