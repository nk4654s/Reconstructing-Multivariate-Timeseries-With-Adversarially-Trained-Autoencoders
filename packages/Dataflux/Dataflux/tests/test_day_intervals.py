"""Dataflux/data/utils.py"""

import unittest
from datetime import datetime, timezone
from Dataflux.helper import parse_datestring, get_day_list, daterange


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_date_parse_with_utc(self):
        date_str = "2025-01-22T00:00:00Z"

        date_obj = parse_datestring(date_str)
        self.assertEqual(date_obj, datetime(2025, 1, 22, tzinfo=timezone.utc))

    def test_date_parse_without_utc(self):
        date_str = "2025-01-22T01:00:00"

        date_obj = parse_datestring(date_str)
        self.assertEqual(date_obj, datetime(2025, 1, 22, tzinfo=timezone.utc))

    def test_date_parse_without_utc_force(self):
        date_str = "2025-01-22T00:00:00"

        date_obj = parse_datestring(date_str, force_utc=False)
        self.assertEqual(date_obj, datetime(2025, 1, 22))

    def test_daterange_without_utc(self):
        date_start = parse_datestring("2025-01-20", force_utc=False)
        date_end = parse_datestring("2025-01-22", force_utc=False)

        date_list = [dt for dt in daterange(date_start, date_end)]
        expected = [
            (datetime(2025, 1, 20, 0, 0), datetime(2025, 1, 20, 23, 59, 59, 999999)),
            (datetime(2025, 1, 21, 0, 0), datetime(2025, 1, 21, 23, 59, 59, 999999)),
        ]

        self.assertEqual(expected, date_list)

    def test_daterange_with_utc(self):
        date_start = parse_datestring("2025-01-20T00:00:00Z")
        date_end = parse_datestring("2025-01-22T00:00:00Z")

        date_list = [dt for dt in daterange(date_start, date_end)]
        expected = [
            (parse_datestring("2025-01-20T00:00:00Z"), datetime(2025, 1, 20, 23, 59, 59, 999999, tzinfo=timezone.utc)),
            (parse_datestring("2025-01-21T00:00:00Z"), datetime(2025, 1, 21, 23, 59, 59, 999999, tzinfo=timezone.utc)),
        ]

        self.assertEqual(expected, date_list)

    def test_get_day_list(self):
        """Test get_day_list function"""
        date_start = parse_datestring("2025-01-20", force_utc=False)
        date_end = parse_datestring("2025-01-22", force_utc=False)

        day_list = get_day_list(date_start, date_end)
        expected = [
            parse_datestring("2025-01-20", force_utc=False),
            parse_datestring("2025-01-21", force_utc=False),
            parse_datestring("2025-01-22", force_utc=False),
        ]

        self.assertEqual(expected, day_list)
