"""Dataflux/data/header.py"""

import unittest

from Dataflux.data import header


def relative_complement(header1, header2):
    header1.relative_complement(header2)


class TestHeader(unittest.TestCase):
    def setUp(self):
        self.header1 = header.Header()

    def relative_complement_error(self):
        self.header1.values = ['a', 'b', 'c']
        header2 = set(['a', 'c', 'd'])
        self.header1.relative_complement(header2)

    def test_relative_complement_eqality(self):
        """Test Header.relative_complement method with equal headers"""
        self.header1.values = ['a', 'b', 'c']
        header2 = set(['a', 'b', 'c'])

        self.header1.relative_complement(header2)
        self.assertEqual(self.header1.drop, [])

    def test_relative_complement_assertion_error(self):
        """Test Header.relative_complement method"""
        self.assertRaises(AssertionError, self.relative_complement_error)

    def test_relative_complement_drop_values(self):
        """Test Header.relative_complement method with drop values as return"""
        self.header1.values = ['a', 'b', 'c']
        header2 = set(['a', 'c'])

        self.header1.relative_complement(header2)

        self.assertEqual(len(self.header1.drop), 1)
        self.assertTrue('b' in self.header1.drop)
